import base64
import json

import requests
import streamlit as st
from streamlit_cookies_controller import CookieController

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
# Instantiate globally for the script run
cookie_controller = CookieController(key="weld_cookies_new")
st.session_state.cookie_controller = cookie_controller

# -------------------------------------------------------------------
# 1. Initialize Navigation early to preserve URL state across reruns!
# -------------------------------------------------------------------
from frontend.admin_ui import show_admin_page
from frontend.auth_ui import logout, show_auth_page
from frontend.chat_ui import show_chat_page
from frontend.monitoring_ui import show_monitoring_page

# Define Pages
chat_page = st.Page(show_chat_page, title="Chatbot", icon="💬", url_path="chat", default=True)
monitoring_page = st.Page(show_monitoring_page, title="RAG Monitoring", icon="📊", url_path="monitoring")
admin_page = st.Page(show_admin_page, title="System Settings", icon="⚙️", url_path="settings")

nav_structure = {
    "User": [chat_page],
    "Admin": [monitoring_page, admin_page]
}
pg = st.navigation(nav_structure)

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

api = st.session_state.api_session

# Restore token from browser cookie on rerun/new session.
if cookie_controller and not st.session_state.force_logged_out:
    try:
        cookie_token = _normalize_token(cookie_controller.get("weld_access_token"))
        if cookie_token and not st.session_state.get("access_token"):
            st.session_state.access_token = cookie_token
        elif (
            not st.session_state.get("access_token")
            and not st.session_state.cookie_restore_attempted
        ):
            # The component can return empty on its first render cycle; retry once.
            st.session_state.cookie_restore_attempted = True
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
        elif response.status_code in (401, 403):
            st.session_state.access_token = None
            if cookie_controller:
                try:
                    cookie_controller.remove("weld_access_token", path="/", same_site="lax")
                except Exception:
                    pass
    except Exception:
        # Keep cookie/token on transient backend/network failures.
        pass

user = st.session_state.user or {}

if not st.session_state.authenticated:
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
    from frontend.auth_ui import show_auth_page
    show_auth_page(api, API_URL)
else:
    # Render Custom Sidebar Elements above/below navigation
    with st.sidebar:
        st.title("🤖 WELDBOT v4.0")
        st.write(f"Logged in as: **{user.get('username', 'unknown')}** ({user.get('role', 'user')})")

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

    # Hide Admin pages from sidebar for normal users via CSS if needed, 
    # but since Streamlit st.navigation doesn't support dynamic hiding after init without rerun,
    # we enforce access control at the page level or just accept they see the menu.
    # Actually, we can restrict by checking user role before pg.run():
    if (pg.url_path in ["monitoring", "settings"]) and user.get("role") != "admin":
        st.error("관리자 권한이 필요합니다.")
        st.stop()

    # Run the selected page
    pg.run()
