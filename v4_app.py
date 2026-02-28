import base64
import json
import time

import requests
import streamlit as st
from streamlit_cookies_controller import CookieController
import extra_streamlit_components as stx

st.set_page_config(
    page_title="WELDBOT v4.0",
    page_icon="🤖",
    layout="wide",
)

API_URL = "http://localhost:8000"


def _normalize_token(raw_token) -> str | None:
    if raw_token is None:
        return None

    token = str(raw_token).strip()
    if not token:
        return None

    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        token = token[1:-1].strip()

    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    return token or None


def _decode_jwt_payload(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        return {}


def _read_context_cookie_token() -> str | None:
    try:
        cookies = st.context.cookies
    except Exception:
        return None

    if not cookies:
        return None

    return _normalize_token(cookies.get("weld_access_token"))


def _sync_access_token_cookie(controller: CookieController | None, token: str | None) -> None:
    normalized = _normalize_token(token)
    if not controller or not normalized:
        return
    try:
        controller.set(
            "weld_access_token",
            normalized,
            path="/",
            max_age=60 * 60 * 24 * 7,
            same_site="lax",
        )
    except Exception:
        pass


def _read_backup_cookie_token() -> str | None:
    manager = st.session_state.get("cookie_manager_auth")
    if manager is None:
        try:
            manager = stx.CookieManager(key="weld_cookie_manager_auth")
            st.session_state.cookie_manager_auth = manager
        except Exception:
            manager = None
    if not manager:
        return None

    try:
        return _normalize_token(manager.get(cookie="weld_access_token"))
    except Exception:
        return None


def _sync_backup_access_token_cookie(token: str | None) -> None:
    normalized = _normalize_token(token)
    if not normalized:
        return

    manager = st.session_state.get("cookie_manager_auth")
    if manager is None:
        try:
            manager = stx.CookieManager(key="weld_cookie_manager_auth")
            st.session_state.cookie_manager_auth = manager
        except Exception:
            manager = None
    if not manager:
        return

    try:
        manager.set(
            cookie="weld_access_token",
            val=normalized,
            path="/",
            max_age=60 * 60 * 24 * 7,
            same_site="lax",
        )
    except Exception:
        pass


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "api_session" not in st.session_state:
    st.session_state.api_session = requests.Session()
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "force_logged_out" not in st.session_state:
    st.session_state.force_logged_out = False
if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = "💬 Chatbot"
if "public_route" not in st.session_state:
    st.session_state.public_route = "main"
if "auth_route" not in st.session_state:
    st.session_state.auth_route = "home"
# Instantiate globally for the script run
cookie_controller = CookieController(key="weld_cookies_new")
st.session_state.cookie_controller = cookie_controller

# -------------------------------------------------------------------
# 1. Initialize Navigation early to preserve URL state across reruns!
# -------------------------------------------------------------------
from frontend.admin_ui import show_admin_page
from frontend.auth_ui import logout
from frontend.chat import show_chat_page
from frontend.login import show_login_page
from frontend.main_page import show_main_page
from frontend.main_page_logout import show_main_page_logout
from frontend.monitoring_ui import show_monitoring_page
from frontend.signup import show_signup_page


def _set_public_route(route: str) -> None:
    st.session_state.public_route = route
    st.rerun()


def _set_auth_route(route: str) -> None:
    st.session_state.auth_route = route
    st.rerun()

# -------------------------------------------------------------------
# 2. Handle Cookies & Auth State
# -------------------------------------------------------------------
import time
if "cookie_initialized" not in st.session_state:
    # Streamlit Custom Component is async JS. Give it time to flush cookie to python backend.
    time.sleep(0.3)
    # The component needs to render at least once
    st.session_state.cookie_initialized = True
    st.rerun()

if "cookie_restore_attempted" not in st.session_state:
    st.session_state.cookie_restore_attempted = False
if "cookie_restore_attempt_count" not in st.session_state:
    st.session_state.cookie_restore_attempt_count = 0

api = st.session_state.api_session

# Restore token from browser cookie on rerun/new session.
if not st.session_state.force_logged_out:
    try:
        # 1) Read request cookie directly (most reliable on hard refresh).
        cookie_token = _read_context_cookie_token()

        # 2) Fallback to component-managed cookie cache.
        if not cookie_token and cookie_controller:
            # Always refresh component cache before reading; otherwise the first
            # empty snapshot can stick in session_state across reruns.
            cookie_controller.refresh()
            cookie_token = _normalize_token(cookie_controller.get("weld_access_token"))

        # 3) Backup cookie manager fallback.
        if not cookie_token:
            cookie_token = _read_backup_cookie_token()

        if cookie_token and not st.session_state.get("access_token"):
            st.session_state.access_token = cookie_token
            st.session_state.cookie_restore_attempted = False
            st.session_state.cookie_restore_attempt_count = 0
        elif (
            not st.session_state.get("access_token")
            and st.session_state.cookie_restore_attempt_count < 3
        ):
            # Custom cookie component may return empty on early cycles.
            st.session_state.cookie_restore_attempt_count += 1
            st.session_state.cookie_restore_attempted = True
            time.sleep(0.15)
            st.rerun()
    except Exception:
        pass

if not st.session_state.authenticated and st.session_state.access_token:
    token = _normalize_token(st.session_state.access_token)
    st.session_state.access_token = token
    try:
        response = api.get(
            f"{API_URL}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        if response.status_code == 200:
            st.session_state.user = response.json()
            st.session_state.authenticated = True
            st.session_state.cookie_restore_attempted = False
            st.session_state.cookie_restore_attempt_count = 0
            # Keep browser cookie in sync while authenticated.
            _sync_access_token_cookie(cookie_controller, token)
            _sync_backup_access_token_cookie(token)
        elif response.status_code in (401, 403):
            st.session_state.access_token = None
            if cookie_controller:
                try:
                    cookie_controller.remove("weld_access_token", path="/", same_site="lax")
                except Exception:
                    pass
            backup_manager = st.session_state.get("cookie_manager_auth")
            if backup_manager:
                try:
                    backup_manager.delete("weld_access_token")
                except Exception:
                    pass
    except Exception:
        # Keep cookie/token on transient backend/network failures.
        pass

user = st.session_state.user or {}

# If session is authenticated but request cookie is missing (or stale),
# rewrite cookie from in-memory token to prevent logout on next hard refresh.
if st.session_state.authenticated and st.session_state.access_token:
    context_cookie = _read_context_cookie_token()
    memory_token = _normalize_token(st.session_state.access_token)
    if memory_token and context_cookie != memory_token:
        _sync_access_token_cookie(cookie_controller, memory_token)
        _sync_backup_access_token_cookie(memory_token)

if not st.session_state.authenticated:
    requested_public = st.query_params.get("public")
    if requested_public in {"main", "login", "signup"}:
        st.session_state.public_route = requested_public
        st.query_params.clear()

    # Hide the sidebar if unauthenticated
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {display: none;}
            [data-testid="stSidebar"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    public_route = st.session_state.get("public_route", "main")
    if public_route == "login":
        show_login_page(
            api=api,
            api_url=API_URL,
            on_back=lambda: _set_public_route("main"),
            on_signup=lambda: _set_public_route("signup"),
        )
    elif public_route == "signup":
        show_signup_page(
            api=api,
            api_url=API_URL,
            on_login=lambda: _set_public_route("login"),
        )
    else:
        show_main_page(
            on_login=lambda: _set_public_route("login"),
            on_signup=lambda: _set_public_route("signup"),
        )
    st.stop()
else:
    role = user.get("role", "user")
    route = st.session_state.get("auth_route", "home")
    if route in {"monitoring", "settings"} and role != "admin":
        st.session_state.auth_route = "home"
        route = "home"

    # Render Custom Sidebar Elements above/below navigation
    with st.sidebar:
        st.title("🤖 WELDBOT v4.0")
        st.write(f"Logged in as: **{user.get('username', 'unknown')}** ({role})")

        st.subheader("Navigation")
        if st.button("🏠 Home", width="stretch", key="side_home"):
            _set_auth_route("home")
        if st.button("💬 Chatbot", width="stretch", key="side_chat"):
            _set_auth_route("chat")
        if role == "admin":
            if st.button("📊 RAG Monitoring", width="stretch", key="side_monitoring"):
                _set_auth_route("monitoring")
            if st.button("⚙️ System Settings", width="stretch", key="side_settings"):
                _set_auth_route("settings")

        token = st.session_state.access_token
        with st.expander("Auth Token"):
            st.code(token or "(token not found)", language="text")
            payload = _decode_jwt_payload(token) if token else {}
            if payload:
                st.json(payload)
                token_id = payload.get("id") or payload.get("sub") or payload.get("username")
                if token_id:
                    st.write(f"Token ID: `{token_id}`")

        st.divider()
        if st.button("Logout", width="stretch"):
            logout(api, API_URL)

    if route in {"monitoring", "settings"} and role != "admin":
        st.error("관리자 권한이 필요합니다.")
        st.stop()

    if route == "chat":
        show_chat_page()
    elif route == "monitoring":
        show_monitoring_page()
    elif route == "settings":
        show_admin_page()
    else:
        show_main_page_logout(
            on_logout=lambda: logout(api, API_URL),
            on_start_chat=lambda: _set_auth_route("chat"),
        )
