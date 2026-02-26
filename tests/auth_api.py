import os
import secrets
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import postgresql_connection as pc
import psycopg2
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="Auth API")

pc.load_env_file(Path(__file__).resolve().parent / ".env")

SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
FRONTEND_DEFAULT_REDIRECT_URI = os.getenv("FRONTEND_DEFAULT_REDIRECT_URI", "http://127.0.0.1:8501")
COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "access_token")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in {"1", "true", "yes", "on"}
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

DB_CONN_KWARGS = pc.get_connection_kwargs()

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)
oauth.register(
    name="kakao",
    client_id=os.getenv("KAKAO_CLIENT_ID"),
    client_secret=os.getenv("KAKAO_CLIENT_SECRET"),
    authorize_url="https://kauth.kakao.com/oauth/authorize",
    access_token_url="https://kauth.kakao.com/oauth/token",
    api_base_url="https://kapi.kakao.com",
    client_kwargs={"scope": "profile_nickname account_email"},
)

oauth_exchange_codes: dict[str, dict[str, object]] = {}


@contextmanager
def get_db_connection():
    with pc.open_optional_ssh_tunnel() as tunnel:
        effective_conn_kwargs = dict(DB_CONN_KWARGS)
        if "conninfo" in effective_conn_kwargs and "dsn" not in effective_conn_kwargs:
            effective_conn_kwargs["dsn"] = effective_conn_kwargs.pop("conninfo")
        if tunnel:
            effective_conn_kwargs["host"] = tunnel["forward_host"]
            effective_conn_kwargs["port"] = tunnel["forward_port"]
        with psycopg2.connect(**effective_conn_kwargs) as conn:
            yield conn


def ensure_user_table() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    name TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            # Backfill columns for environments that created an older users schema.
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT")
            cur.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
            )
        conn.commit()


@app.on_event("startup")
def on_startup() -> None:
    ensure_user_table()


def db_get_user(username: str) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, password_hash, email, name FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "username": row[0],
        "password_hash": row[1],
        "email": row[2],
        "name": row[3],
    }


def db_create_user(username: str, password_hash: str, email: str | None = None, name: str | None = None) -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, password_hash, email, name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (username) DO NOTHING
                """,
                (username, password_hash, email, name),
            )
            created = cur.rowcount == 1
        conn.commit()
    return created


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    user = db_get_user(username)
    if not user:
        return None
    if not user["password_hash"]:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def set_auth_cookie(response, access_token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def clear_auth_cookie(response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/")


def is_provider_configured(provider: str) -> bool:
    if provider == "google":
        return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))
    if provider == "kakao":
        return bool(os.getenv("KAKAO_CLIENT_ID"))
    return False


def get_provider_client(provider: str):
    if provider not in {"google", "kakao"}:
        raise HTTPException(status_code=404, detail="Unsupported provider")
    if not is_provider_configured(provider):
        raise HTTPException(
            status_code=500,
            detail=f"{provider} OAuth is not configured. Check env vars.",
        )
    return oauth.create_client(provider)


def build_redirect_with_code(frontend_redirect_uri: str, auth_code: str) -> str:
    query = urlencode({"auth_code": auth_code})
    sep = "&" if "?" in frontend_redirect_uri else "?"
    return f"{frontend_redirect_uri}{sep}{query}"


def issue_oauth_exchange_code(username: str) -> str:
    code = secrets.token_urlsafe(32)
    expire_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    oauth_exchange_codes[code] = {"username": username, "expire_at": expire_at}
    return code


def consume_oauth_exchange_code(code: str) -> str:
    payload = oauth_exchange_codes.pop(code, None)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or already used auth_code")
    expire_at = payload["expire_at"]
    if not isinstance(expire_at, datetime) or expire_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Expired auth_code")
    username = payload["username"]
    if not isinstance(username, str):
        raise HTTPException(status_code=401, detail="Invalid auth_code payload")
    return username


def get_current_user(request: Request, bearer_token: str | None = Depends(oauth2_scheme)):
    token = bearer_token or request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db_get_user(username)
    if user is None:
        raise credentials_exception
    return user


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    response = JSONResponse(content={"message": "login success"})
    set_auth_cookie(response, access_token)
    return response


@app.post("/auth/signup")
def signup(username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    created = db_create_user(username=username, password_hash=pwd_context.hash(password))
    if not created:
        raise HTTPException(status_code=409, detail="Username already exists")

    return {"message": "signup success"}


@app.post("/auth/logout")
def logout():
    response = JSONResponse(content={"message": "logout success"})
    clear_auth_cookie(response)
    return response


@app.post("/auth/oauth/exchange")
def oauth_exchange(code: str = Form(...)):
    username = consume_oauth_exchange_code(code)
    access_token = create_access_token(
        data={"sub": username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    response = JSONResponse(content={"message": "oauth exchange success"})
    set_auth_cookie(response, access_token)
    return response


@app.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}


@app.get("/auth/oauth/{provider}/login")
async def oauth_login(
    provider: str,
    request: Request,
    frontend_redirect_uri: str = Query(default=FRONTEND_DEFAULT_REDIRECT_URI),
):
    client = get_provider_client(provider)
    request.session["frontend_redirect_uri"] = frontend_redirect_uri
    redirect_uri = request.url_for("oauth_callback", provider=provider)
    return await client.authorize_redirect(request, redirect_uri)


@app.get("/auth/oauth/{provider}/callback")
async def oauth_callback(provider: str, request: Request):
    client = get_provider_client(provider)
    frontend_redirect_uri = request.session.pop(
        "frontend_redirect_uri",
        FRONTEND_DEFAULT_REDIRECT_URI,
    )

    try:
        token = await client.authorize_access_token(request)
    except OAuthError as exc:
        raise HTTPException(status_code=401, detail=f"OAuth failed: {exc.error}")

    if provider == "google":
        userinfo_response = await client.get("userinfo", token=token)
        userinfo = userinfo_response.json()
        provider_user_id = userinfo.get("sub")
        email = userinfo.get("email")
        name = userinfo.get("name") or email or "google-user"
    else:
        userinfo_response = await client.get("/v2/user/me", token=token)
        userinfo = userinfo_response.json()
        provider_user_id = str(userinfo.get("id")) if userinfo.get("id") is not None else None
        account = userinfo.get("kakao_account", {}) or {}
        profile = account.get("profile", {}) or {}
        email = account.get("email")
        name = profile.get("nickname") or email or "kakao-user"

    if not provider_user_id:
        raise HTTPException(status_code=400, detail="Failed to read user info from provider")

    internal_username = f"{provider}:{provider_user_id}"
    db_create_user(
        username=internal_username,
        password_hash="",
        email=email,
        name=name,
    )

    auth_code = issue_oauth_exchange_code(internal_username)
    return RedirectResponse(url=build_redirect_with_code(frontend_redirect_uri, auth_code))
