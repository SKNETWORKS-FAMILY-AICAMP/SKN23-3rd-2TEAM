import os
import secrets
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import psycopg2
from typing import Optional

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, status, Response, Cookie
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

# Use the core database module for connection and tunneling
from app.core.database import open_optional_ssh_tunnel, get_connection_kwargs, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["Authentication"])

SECRET_KEY = os.getenv("SECRET_KEY", "weld-bot-v4-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
FRONTEND_DEFAULT_REDIRECT_URI = os.getenv("FRONTEND_DEFAULT_REDIRECT_URI", "http://localhost:8501")
COOKIE_NAME = "weld_auth_token"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in {"1", "true", "yes", "on"}
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# ---------------------------------------------------------
# OAuth Setup
# ---------------------------------------------------------
oauth = OAuth()
if os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if os.getenv("KAKAO_CLIENT_ID"):
    # Kakao sometimes needs client_secret too, but let's register if client_id exists
    oauth.register(
        name="kakao",
        client_id=os.getenv("KAKAO_CLIENT_ID"),
        client_secret=os.getenv("KAKAO_CLIENT_SECRET", ""),
        authorize_url="https://kauth.kakao.com/oauth/authorize",
        access_token_url="https://kauth.kakao.com/oauth/token",
        api_base_url="https://kapi.kakao.com",
        client_kwargs={"scope": "profile_nickname account_email"},
    )

oauth_exchange_codes: dict[str, dict[str, object]] = {}

# ---------------------------------------------------------
# Database Utils
# ---------------------------------------------------------
@contextmanager
def get_db_connection():
    with open_optional_ssh_tunnel() as tunnel:
        conn_args = get_connection_kwargs()
        if tunnel:
            conn_args["host"] = tunnel["host"]
            conn_args["port"] = tunnel["port"]
        with psycopg2.connect(**conn_args) as conn:
            yield conn

def db_get_oauth_user(username: str) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash, role FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "username": row[1],
        "password": row[2],
        "role": row[3],
    }

def db_create_oauth_user(username: str, role: str = "user") -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, %s, %s)
                ON CONFLICT (username) DO NOTHING
                """,
                (username, "", role),
            )
            created = cur.rowcount == 1
        conn.commit()
    return created

# ---------------------------------------------------------
# Auth Utils
# ---------------------------------------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def set_auth_cookie(response: Response, access_token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )

def clear_auth_cookie(response: Response) -> None:
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

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@router.post("/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    user = db_get_oauth_user(username)
    if not user or not user["password"] or not verify_password(password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    access_token = create_access_token(
        data={"sub": user["username"], "id": user["id"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    set_auth_cookie(response, access_token)
    
    return {
        "message": "Login successful",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"]
        }
    }

@router.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...), admin_code: Optional[str] = Form(None)):
    from app.core.config import ADMIN_SECRET_KEY
    
    role = "user"
    if admin_code and admin_code == ADMIN_SECRET_KEY:
        role = "admin"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Username already exists")
            
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                (username, hash_password(password), role)
            )
        conn.commit()
    return {"message": "Signup successful"}

@router.post("/logout")
async def logout(response: Response):
    clear_auth_cookie(response)
    return {"message": "Logout successful"}

@router.get("/me")
async def get_me(request: Request, weld_auth_token: Optional[str] = Cookie(None)):
    token = weld_auth_token or request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "id": payload.get("id"),
            "username": payload.get("sub"),
            "role": payload.get("role")
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.get("/oauth/{provider}/login")
async def oauth_login(
    provider: str,
    request: Request,
    frontend_redirect_uri: str = Query(default=FRONTEND_DEFAULT_REDIRECT_URI),
):
    client = get_provider_client(provider)
    request.session["frontend_redirect_uri"] = frontend_redirect_uri
    redirect_uri = request.url_for("oauth_callback", provider=provider)
    return await client.authorize_redirect(request, redirect_uri)

@router.get("/oauth/{provider}/callback")
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
    else:
        userinfo_response = await client.get("/v2/user/me", token=token)
        userinfo = userinfo_response.json()
        provider_user_id = str(userinfo.get("id")) if userinfo.get("id") is not None else None

    if not provider_user_id:
        raise HTTPException(status_code=400, detail="Failed to read user info from provider")

    internal_username = f"{provider}:{provider_user_id}"
    user = db_get_oauth_user(internal_username)
    if not user:
        db_create_oauth_user(username=internal_username)

    auth_code = issue_oauth_exchange_code(internal_username)
    sep = "&" if "?" in frontend_redirect_uri else "?"
    redirect_url = f"{frontend_redirect_uri}{sep}auth_code={auth_code}"
    
    return RedirectResponse(url=redirect_url)

@router.post("/oauth/exchange")
async def oauth_exchange(response: Response, code: str = Form(...)):
    username = consume_oauth_exchange_code(code)
    user = db_get_oauth_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    access_token = create_access_token(
        data={"sub": user["username"], "id": user["id"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    
    set_auth_cookie(response, access_token)
    return {
        "message": "oauth exchange success",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"]
        },
        "weld_auth_token": access_token
    }

def setup_auth_middleware(app):
    from starlette.middleware.sessions import SessionMiddleware
    app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)