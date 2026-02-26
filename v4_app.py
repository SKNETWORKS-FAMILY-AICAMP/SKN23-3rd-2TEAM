import base64
import json

import requests
import streamlit as st

st.set_page_config(
    page_title="WELDBOT v4.0",
    page_icon="🤖",
    layout="wide",
)

API_URL = "http://localhost:8000"


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

api = st.session_state.api_session

if not st.session_state.authenticated and st.session_state.access_token:
    try:
        response = api.get(
            f"{API_URL}/auth/me",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            timeout=5,
        )
        if response.status_code == 200:
            st.session_state.user = response.json()
            st.session_state.authenticated = True
            st.rerun()
    except Exception:
        pass

if not st.session_state.authenticated:
    from frontend.auth_ui import show_auth_page

    show_auth_page(api, API_URL)
else:
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

        selection = st.radio("Navigation", menu_options)

        st.divider()
        if st.button("Logout", use_container_width=True):
            logout(api, API_URL)

    if selection == "Chatbot":
        show_chat_page()
    elif selection == "Admin Dashboard" and user.get("role") == "admin":
        show_admin_page()
