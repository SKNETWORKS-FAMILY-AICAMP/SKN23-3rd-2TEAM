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
if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = "Chatbot"
if "cookie_controller" not in st.session_state:
    try:
        st.session_state.cookie_controller = CookieController(key="weldbot_auth_cookie_controller")
    except Exception:
        st.session_state.cookie_controller = None
if "cookie_restore_attempted" not in st.session_state:
    st.session_state.cookie_restore_attempted = False

api = st.session_state.api_session
cookie_controller = st.session_state.get("cookie_controller")

# Restore token from browser cookie on rerun/new session.
if cookie_controller:
    try:
        cookie_controller.refresh()
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

if not st.session_state.authenticated:
    from frontend.auth_ui import show_auth_page

    show_auth_page(api, API_URL)

if not st.session_state.authenticated:
    st.stop()

from frontend.admin_ui import show_admin_page
from frontend.auth_ui import logout
from frontend.chat_ui import show_chat_page

user = st.session_state.user

with st.sidebar:
    st.title("🤖 WELDBOT v4.0")
    st.write(f"Logged in as: **{user['username']}** ({user['role']})")

    token = st.session_state.access_token
    with st.expander("Auth Token"):
        st.code(token or "(token not found)", language="text")
        payload = _decode_jwt_payload(token) if token else {}
        if payload:
            st.json(payload)
            token_id = payload.get("id") or payload.get("sub") or payload.get("username")
            if token_id:
                st.write(f"Token ID: `{token_id}`")

    menu_options = ["Chatbot"]
    if user.get("role") == "admin":
        menu_options.append("Admin Dashboard")

    if st.session_state.nav_selection not in menu_options:
        st.session_state.nav_selection = "Chatbot"
    selection = st.radio(
        "Navigation",
        menu_options,
        index=menu_options.index(st.session_state.nav_selection),
    )
    st.session_state.nav_selection = selection

    st.divider()
    if st.button("Logout", width="stretch"):
        logout(api, API_URL)

if selection == "Chatbot":
    show_chat_page()
elif selection == "Admin Dashboard" and user.get("role") == "admin":
    show_admin_page()
