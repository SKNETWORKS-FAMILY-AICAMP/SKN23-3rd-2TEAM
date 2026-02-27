import requests
import streamlit as st


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


def _save_access_token_cookie(token: str) -> None:
    controller = st.session_state.get("cookie_controller")
    normalized_token = _normalize_token(token)
    if not controller or not normalized_token:
        return
    try:
        # Persist for 7 days; token validity is still enforced by backend exp.
        controller.set(
            "weld_access_token",
            normalized_token,
            path="/",
            max_age=60 * 60 * 24 * 7,
            same_site="lax",
        )
    except Exception:
        pass


def _clear_access_token_cookie() -> None:
    controller = st.session_state.get("cookie_controller")
    if not controller:
        return
    try:
        controller.remove("weld_access_token", path="/", same_site="lax")
    except Exception:
        pass


def show_auth_page(api, API_URL):
    st.title("WELDBOT v4.0 Access Control")
    st.caption("Token cache mode (session_state + browser cookie)")

    if "access_token" not in st.session_state:
        st.session_state.access_token = None

    # Already logged-in users should not see the login form again.
    if st.session_state.get("authenticated") and st.session_state.get("access_token"):
        user = st.session_state.get("user") or {}
        st.success(f"관리자: {user.get('username', 'unknown')}")
        st.info("왼쪽/상단의 이동 버튼으로 Chat UI로 돌아가세요.")
        return

    # If a cached token exists, validate it and immediately return to the main app route.
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
                return
            elif me_resp.status_code in (401, 403):
                st.session_state.access_token = None
                _clear_access_token_cookie()
        except Exception:
            # Keep token on transient backend/network failures.
            pass

    auth_code = st.query_params.get("auth_code")
    if auth_code and not st.session_state.get("authenticated"):
        try:
            response = api.post(f"{API_URL}/auth/oauth/exchange", data={"code": auth_code}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.user = data.get("user")
                st.session_state.access_token = _normalize_token(data.get("weld_auth_token"))
                st.session_state.authenticated = bool(st.session_state.user and st.session_state.access_token)
                st.session_state.force_logged_out = False
                if st.session_state.access_token:
                    _save_access_token_cookie(st.session_state.access_token)
                    st.session_state.cookie_restore_attempted = False
                st.query_params.clear()
                return
            else:
                st.error("소셜 로그인 인증에 실패했습니다.")
        except Exception as exc:
            st.error(f"소셜 로그인 연결 오류: {exc}")

    tab1, tab2 = st.tabs(["로그인", "회원가입"])

    with tab1:
        st.subheader("Login")
        login_id = st.text_input("Username", key="login_id")
        login_pw = st.text_input("Password", type="password", key="login_pw")

        if st.button("Login", use_container_width=True):
            try:
                response = api.post(
                    f"{API_URL}/auth/login",
                    data={"username": login_id, "password": login_pw},
                    timeout=10,
                )
            except Exception as exc:
                st.error(f"백엔드 연결 실패: {exc}")
                return

            if response.status_code != 200:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
                return

            data = response.json()
            token = data.get("weld_auth_token")
            if not token:
                st.error("로그인 응답에 토큰이 없습니다.")
                return

            st.session_state.user = data.get("user")
            st.session_state.access_token = _normalize_token(token)
            st.session_state.authenticated = True
            st.session_state.force_logged_out = False
            _save_access_token_cookie(st.session_state.access_token)
            st.session_state.cookie_restore_attempted = False
            st.success(f"{login_id}님 환영합니다.")
            import time
            time.sleep(0.5)
            st.rerun()

        st.markdown("---")
        st.markdown("### 소셜 로그인")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(
                f'<a href="{API_URL}/auth/oauth/google/login?frontend_redirect_uri=http://localhost:8501" target="_self"><button style="width:100%">Google Login</button></a>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f'<a href="{API_URL}/auth/oauth/kakao/login?frontend_redirect_uri=http://localhost:8501" target="_self"><button style="width:100%">Kakao Login</button></a>',
                unsafe_allow_html=True,
            )

    with tab2:
        st.subheader("Sign Up")
        new_id = st.text_input("New Username", key="new_id")
        new_pw = st.text_input("New Password", type="password", key="new_pw")
        confirm_pw = st.text_input("Confirm Password", type="password", key="confirm_pw")
        admin_code = st.text_input("관리자 가입 코드 (선택)", type="password", key="admin_code")

        if st.button("Sign Up", use_container_width=True):
            if not new_id or not new_pw:
                st.warning("아이디와 비밀번호를 모두 입력해 주세요.")
            elif new_pw != confirm_pw:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                try:
                    response = api.post(
                        f"{API_URL}/auth/signup",
                        data={"username": new_id, "password": new_pw, "admin_code": admin_code},
                        timeout=10,
                    )
                    if response.status_code == 200:
                        st.success("회원가입이 완료되었습니다. 로그인 탭에서 로그인해 주세요.")
                    else:
                        st.error("이미 존재하는 아이디이거나 회원가입 중 오류가 발생했습니다.")
                except Exception as exc:
                    st.error(f"백엔드 연결 실패: {exc}")


def logout(api, API_URL):
    token = st.session_state.get("access_token")
    headers = {"Authorization": f"Bearer {token}"} if token else None

    try:
        api.post(f"{API_URL}/auth/logout", headers=headers, timeout=5)
    except Exception:
        pass

    st.session_state.user = None
    st.session_state.authenticated = False
    st.session_state.access_token = None
    st.session_state.force_logged_out = True
    _clear_access_token_cookie()

    st.session_state.api_session = requests.Session()
    import time
    time.sleep(0.5)
    st.rerun()
