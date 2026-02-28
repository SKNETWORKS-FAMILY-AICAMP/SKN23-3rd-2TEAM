import re

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


def validate_id(user_id):
    if not user_id.strip():
        return "아이디를 입력해주세요."
    if len(user_id) < 4:
        return "아이디는 4자 이상 입력해주세요."
    if len(user_id) > 12:
        return "아이디는 12자 이하로 입력해주세요."
    if re.search(r"[^a-zA-Z0-9]", user_id):
        return "아이디는 영문과 숫자만 사용 가능합니다. (특수문자 불가)"
    return None


def validate_password(pw):
    # 비밀번호 정책 제한 없음
    return None


def show_login_page(api, api_url, on_back, on_signup):
    st.markdown(
        """
        <style>
        :root {
            --nav-h: 64px;
            --line: rgba(120, 190, 220, 0.20);
            --primary: #ff6a3d;
            --secondary: #2ec5ff;
        }

        html, body { margin: 0 !important; padding: 0 !important; }
        [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
        .stApp { background-color: #f2f2f2; }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            padding-top: 0 !important; margin-top: 0 !important;
        }

        .block-container { max-width: 100% !important; padding: 0 !important; }

        .nav {
            display: flex; align-items: center;
            height: var(--nav-h); position: fixed;
            top: 0; left: 0; right: 0;
            padding: 0 32px 0 24px;
            background: #000; border-bottom: 1px solid var(--line); z-index: 9998;
        }

        .logo-btn {
            display: flex; align-items: center; gap: 10px;
            text-decoration: none !important; cursor: pointer;
        }
        .logo-btn:hover .logo-name { color: #2ec5ff; }

        .logo-dot {
            width: 14px; height: 14px; border-radius: 50%;
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            box-shadow: 0 0 20px rgba(46,197,255,0.8); flex-shrink: 0;
        }

        .logo-name {
            font-size: 0.95rem; font-weight: 600; color: #dbf5ff;
            letter-spacing: 0.08em; transition: color 0.15s;
        }

        .stForm {
            background-color: #fff; border-radius: 0.5rem;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
            padding: 2rem 1.5rem !important; border: none !important;
        }

        .stTextInput label {
            font-size: 14px !important; font-weight: 600 !important; color: #111 !important;
        }
        .stTextInput input {
            border: 1.5px solid #d1d5db !important; border-radius: 8px !important;
            font-size: 15px !important; background: #f2f2f2 !important; color: #111 !important;
        }
        .stTextInput input:focus {
            box-shadow: 0 0 0 2px rgba(0,0,0,0.08) !important; border-color: black !important;
        }

        [data-testid="stFormSubmitButton"] > button {
            background: #000 !important; color: #fff !important;
            border: 1px solid #000 !important; width: 100% !important;
            border-radius: 6px !important; height: 44px !important;
            font-size: 15px !important; font-weight: 600 !important;
        }
        [data-testid="stFormSubmitButton"] > button:hover {
            background: #111 !important; color: #fff !important;
        }

        h1 { text-align: center; margin-bottom: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="nav">
            <a class="logo-btn" href="/?public=main" target="_self">
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
            st.markdown(
                "<h1 style='text-align:center; margin-bottom:2rem; transform: translateX(20px);'>로그인</h1>",
                unsafe_allow_html=True,
            )
            user_id = st.text_input("아이디", placeholder="영문+숫자, 4~12자")
            password = st.text_input("비밀번호", placeholder="비밀번호", type="password")

            st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

            btn1, btn2 = st.columns(2)
            with btn1:
                login_clicked = st.form_submit_button("로그인", use_container_width=True)
            with btn2:
                signup_clicked = st.form_submit_button("회원가입", use_container_width=True)

            st.markdown(
                """
                <div style='text-align:center; margin-top:1rem;'>
                    <a href='#' style='font-size:0.75rem; color:#6b7280; text-decoration:underline;'>비밀번호 찾기</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if login_clicked:
            id_err = validate_id(user_id)
            pw_err = validate_password(password)

            if id_err:
                st.error(f"⚠ {id_err}")
            elif pw_err:
                st.error(f"⚠ {pw_err}")
            else:
                try:
                    response = api.post(
                        f"{api_url}/auth/login",
                        data={"username": user_id, "password": password},
                        timeout=10,
                    )
                except Exception as exc:
                    st.error(f"백엔드 연결 실패: {exc}")
                    return

                if response.status_code != 200:
                    st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
                else:
                    data = response.json()
                    token = _normalize_token(data.get("weld_auth_token"))
                    if not token:
                        st.error("로그인 응답에 토큰이 없습니다.")
                        return

                    st.session_state.user = data.get("user")
                    st.session_state.access_token = token
                    st.session_state.authenticated = True
                    st.session_state.force_logged_out = False
                    st.session_state.cookie_restore_attempted = False

                    controller = st.session_state.get("cookie_controller")
                    if controller:
                        try:
                            controller.set(
                                "weld_access_token",
                                token,
                                path="/",
                                max_age=60 * 60 * 24 * 7,
                                same_site="lax",
                            )
                        except Exception:
                            pass

                    st.success(f"✅ 로그인 성공: {user_id}")
                    st.rerun()

        if signup_clicked:
            on_signup()
