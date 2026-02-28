import re

import streamlit as st


def validate_name(name):
    if not name.strip():
        return "이름을 입력해주세요."
    if len(name) > 10:
        return "이름은 10자 이하로 입력해주세요."
    if re.search(r"[^가-힣a-zA-Z]", name):
        return "이름은 한글 또는 영문만 사용 가능합니다."
    return None


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


def show_signup_page(api, api_url, on_login):
    if "signup_checked_id" not in st.session_state:
        st.session_state["signup_checked_id"] = ""

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
      .stApp { background-color: #f2f2f2; }
      #MainMenu, footer { visibility: hidden; }

      [data-testid="stHeader"],
      [data-testid="stToolbar"] { display: none !important; }

      [data-testid="stAppViewContainer"],
      [data-testid="stMain"],
      [data-testid="stMainBlockContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }

      .block-container {
        max-width: 100% !important;
        padding: 0 !important;
      }

      .nav {
        display: flex;
        align-items: center;
        height: var(--nav-h);
        position: fixed;
        top: 0; left: 0; right: 0;
        padding: 0 32px 0 24px;
        background: #000;
        border-bottom: 1px solid var(--line);
        z-index: 9998;
      }

      .logo-btn {
        display: flex;
        align-items: center;
        gap: 10px;
        text-decoration: none !important;
        cursor: pointer;
      }

      .logo-btn:hover .logo-name { color: #2ec5ff; }

      .logo-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--secondary), var(--primary));
        box-shadow: 0 0 20px rgba(46,197,255,0.8);
        flex-shrink: 0;
      }

      .logo-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #dbf5ff;
        letter-spacing: 0.08em;
        transition: color 0.15s;
      }

      .st-key-signup_nav_login {
        position: fixed;
        top: 12px;
        right: 24px;
        width: 110px;
        z-index: 10001;
      }

      .st-key-signup_nav_login button {
        width: 100% !important;
        background: transparent !important;
        color: #fff !important;
        border: 1.5px solid rgba(255,255,255,0.35) !important;
        border-radius: 8px !important;
        font-size: 0.84rem !important;
        font-weight: 700 !important;
        min-height: 36px !important;
      }

      .st-key-signup_nav_login button:hover {
        background: rgba(255,255,255,0.1) !important;
        color: #fff !important;
      }

      .stForm {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.10);
        padding: 40px 36px 36px !important;
        border: none !important;
      }

      .stTextInput label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #111 !important;
      }

      .stTextInput input {
        border: 1.5px solid #d1d5db !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        background: #f2f2f2 !important;
        color: #111 !important;
      }

      .stTextInput input:focus {
        border-color: #000 !important;
        box-shadow: 0 0 0 2px rgba(0,0,0,0.08) !important;
      }

      [data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        background: #000 !important;
        color: #fff !important;
        border: 1px solid #000 !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
      }

      [data-testid="stFormSubmitButton"] > button:hover {
        background: #111 !important;
        color: #fff !important;
      }
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

    if st.button("로그인", key="signup_nav_login", use_container_width=True):
        on_login()

    st.markdown("<div style='height: calc(64px + 3vh);'></div>", unsafe_allow_html=True)

    _, card, _ = st.columns([2.5, 3, 2.5])

    with card:
        with st.form("signup_form", clear_on_submit=False):
            st.markdown(
                "<h2 style='text-align:center; font-weight:700; margin-bottom:24px; transform: translateX(20px);'>회원가입</h2>",
                unsafe_allow_html=True,
            )

            name = st.text_input("이름", placeholder="한글 또는 영문, 10자 이하")

            id_col, dup_col = st.columns([7, 3])
            with id_col:
                user_id = st.text_input("아이디", placeholder="영문+숫자, 4~12자")
            with dup_col:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                check_clicked = st.form_submit_button("중복확인", use_container_width=True)

            password = st.text_input("비밀번호", placeholder="비밀번호", type="password")

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("가입하기", use_container_width=True)

        if check_clicked:
            id_err = validate_id(user_id)
            if id_err:
                st.error(f"⚠ {id_err}")
                st.session_state["signup_checked_id"] = ""
            else:
                st.success("✓ 사용 가능한 형식의 아이디입니다.")
                st.session_state["signup_checked_id"] = user_id

        if submitted and not check_clicked:
            name_err = validate_name(name)
            id_err = validate_id(user_id)
            pw_err = validate_password(password)

            if name_err:
                st.error(f"⚠ {name_err}")
            elif id_err:
                st.error(f"⚠ {id_err}")
                st.session_state["signup_checked_id"] = ""
            elif pw_err:
                st.error(f"⚠ {pw_err}")
            elif st.session_state["signup_checked_id"] != user_id:
                st.warning("⚠ 아이디 중복확인을 먼저 완료해주세요.")
            else:
                try:
                    response = api.post(
                        f"{api_url}/auth/signup",
                        data={"username": user_id, "password": password},
                        timeout=10,
                    )
                except Exception as exc:
                    st.error(f"백엔드 연결 실패: {exc}")
                    return

                if response.status_code == 200:
                    st.session_state["signup_checked_id"] = ""
                    st.success("✅ 회원가입이 완료되었습니다!")
                else:
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    st.error(f"⚠ 회원가입 실패: {detail}")
