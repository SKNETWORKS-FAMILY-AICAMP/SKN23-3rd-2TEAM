import re

import requests
import streamlit as st

from frontend.auth_ui import (
    _clear_access_token_cookie,
    _normalize_token,
    _save_access_token_cookie,
)

API_URL = "http://localhost:8000"


def validate_id(user_id: str) -> str | None:
    if not user_id.strip():
        return "아이디를 입력해주세요."
    if len(user_id) < 4:
        return "아이디는 4자 이상 입력해주세요."
    if len(user_id) > 12:
        return "아이디는 12자 이하로 입력해주세요."
    if re.search(r"[^a-zA-Z0-9]", user_id):
        return "아이디는 영문과 숫자만 사용 가능합니다."
    return None


def validate_password(password: str) -> str | None:
    if not password.strip():
        return "비밀번호를 입력해주세요."
    if len(password) < 8:
        return "비밀번호는 8자 이상 입력해주세요."
    if len(password) > 20:
        return "비밀번호는 20자 이하로 입력해주세요."
    if re.search(r"[^a-zA-Z0-9]", password):
        return "비밀번호는 영문과 숫자만 사용 가능합니다."
    if not re.search(r"[A-Z]", password):
        return "비밀번호에 대문자를 1자 이상 포함해주세요."
    if not re.search(r"[0-9]", password):
        return "비밀번호에 숫자를 1자 이상 포함해주세요."
    return None


def show_login_page() -> None:
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    api = st.session_state.get("api_session")
    if api is None:
        api = requests.Session()
        st.session_state.api_session = api

    # Keep auth flow aligned with auth_ui: validate cached token before showing form.
    cached_token = _normalize_token(st.session_state.get("access_token"))
    st.session_state.access_token = cached_token
    if cached_token and not st.session_state.get("authenticated"):
        try:
            me_resp = api.get(
                f"{API_URL}/auth/me",
                headers={"Authorization": f"Bearer {cached_token}"},
                timeout=5,
            )
            if me_resp.status_code == 200:
                st.session_state.user = me_resp.json()
                st.session_state.authenticated = True
                st.session_state.cookie_restore_attempted = False
                st.markdown("<meta http-equiv='refresh' content='0; url=/'>", unsafe_allow_html=True)
                st.stop()
            if me_resp.status_code in (401, 403):
                st.session_state.access_token = None
                _clear_access_token_cookie()
        except Exception:
            # Keep token on transient backend/network failures.
            pass

    if st.session_state.get("authenticated") and st.session_state.get("access_token"):
        st.markdown("<meta http-equiv='refresh' content='0; url=/'>", unsafe_allow_html=True)
        st.stop()

    st.markdown(
        """
        <style>
        [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
        .stApp { background-color: #f2f2f2; }
        .block-container { max-width: 100% !important; padding: 0 !important; }
        .nav {
            display: flex; align-items: center; height: 64px; position: fixed;
            top: 0; left: 0; right: 0; padding: 0 24px; background: #000; z-index: 9998;
        }
        .logo-btn { display: flex; align-items: center; gap: 10px; text-decoration: none !important; }
        .logo-dot {
            width: 14px; height: 14px; border-radius: 50%;
            background: linear-gradient(135deg, #2ec5ff, #ff6a3d);
            box-shadow: 0 0 20px rgba(46,197,255,0.8);
        }
        .logo-name { font-size: 0.95rem; font-weight: 600; color: #dbf5ff; letter-spacing: 0.08em; }
        .stForm {
            background-color: #fff; border-radius: 8px;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
            padding: 2rem 1.5rem !important; border: none !important;
        }
        [data-testid="stFormSubmitButton"] > button {
            background: #000 !important; color: #fff !important; border: 1px solid #000 !important;
            width: 100% !important; border-radius: 6px !important; height: 44px !important;
            font-size: 15px !important; font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="nav">
            <a class="logo-btn" href="/" target="_self">
                <div class="logo-dot"></div>
                <div class="logo-name">WELDPILOT AI</div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: calc(64px + 10vh);'></div>", unsafe_allow_html=True)
    _, card, _ = st.columns([2.5, 3, 2.5])

    with card:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("<h1 style='text-align:center; margin-bottom:2rem;'>로그인</h1>", unsafe_allow_html=True)
            user_id = st.text_input("아이디", placeholder="영문+숫자, 4~12자")
            password = st.text_input("비밀번호", placeholder="영문 대소문자+숫자, 8~20자", type="password")
            st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
            left, right = st.columns(2)
            with left:
                login_clicked = st.form_submit_button("로그인", use_container_width=True)
            with right:
                signup_clicked = st.form_submit_button("회원가입", use_container_width=True)

        if signup_clicked:
            st.markdown("<meta http-equiv='refresh' content='0; url=/?auth=signup'>", unsafe_allow_html=True)
            st.stop()

        if login_clicked:
            id_err = validate_id(user_id)
            pw_err = validate_password(password)
            if id_err:
                st.error(id_err)
                return
            if pw_err:
                st.error(pw_err)
                return

            try:
                response = api.post(
                    f"{API_URL}/auth/login",
                    data={"username": user_id, "password": password},
                    timeout=10,
                )
            except Exception as exc:
                st.error(f"백엔드 연결 실패: {exc}")
                return

            if response.status_code != 200:
                try:
                    detail = response.json().get("detail")
                except Exception:
                    detail = None
                st.error(detail or "아이디 또는 비밀번호가 올바르지 않습니다.")
                return

            data = response.json()
            token = _normalize_token(data.get("weld_auth_token") or data.get("access_token"))
            if not token:
                st.error("로그인 응답에 토큰이 없습니다.")
                return

            st.session_state.user = data.get("user")
            st.session_state.access_token = token
            st.session_state.authenticated = True
            st.session_state.force_logged_out = False
            st.session_state.cookie_restore_attempted = False
            _save_access_token_cookie(token)
            st.rerun()


show_login_page()
