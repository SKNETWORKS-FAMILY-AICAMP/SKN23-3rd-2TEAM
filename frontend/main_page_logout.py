from pathlib import Path

import streamlit as st

IMAGE_PATH = Path(r"/Users/jy/3rd-2TEAM/SKN23-3rd-2TEAM/frontend/image/streamlit1.png")


def show_main_page_logout(on_logout, on_start_chat):
    if st.query_params.get("action") == "logout":
        st.query_params.clear()
        on_logout()
        return

    user = st.session_state.get("user")
    username = user.get("username") if isinstance(user, dict) else "사용자"

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;500;700&display=swap');

        :root {
            --nav-h: 64px;
            --line: rgba(120, 190, 220, 0.20);
            --primary: #ff6a3d;
            --secondary: #2ec5ff;
        }

        html, body { margin: 0 !important; padding: 0 !important; }

        [data-testid="stHeader"],
        [data-testid="stToolbar"] { display: none !important; }

        .stApp { background-color: #f2f2f2; }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        .block-container {
            max-width: 100%;
            padding: 0 0 2rem 0;
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

        .logo { display: flex; align-items: center; gap: 10px; }

        .logo-dot {
            width: 14px; height: 14px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            box-shadow: 0 0 20px rgba(46,197,255,0.8);
        }

        .logo-name {
            font-size: 0.95rem;
            font-weight: 650;
            color: #dbf5ff;
            letter-spacing: 0.08em;
        }

        .nav-btns {
            display: flex;
            gap: 10px;
            margin-left: auto;
            align-items: center;
        }

        .nav-btns a {
            display: inline-block;
            padding: 7px 18px;
            border-radius: 8px;
            font-size: 0.84rem;
            font-weight: 700;
            text-decoration: none;
            cursor: pointer;
            transition: background 0.15s;
        }

        .btn-user {
            background: #fff;
            color: #000 !important;
            border: 1.5px solid rgba(0,0,0,0.15);
        }

        .btn-logout {
            background: #000;
            color: #fff !important;
            border: 1.5px solid #000;
        }

        .btn-logout:hover { background: #111; }

        .content-spacer { height: calc(var(--nav-h) + 10px); }

        .st-key-home_nav_chat {
            position: fixed;
            top: 15px;
            right: 132px;
            width: 80px;
            z-index: 10020;
        }
        .st-key-home_nav_logout {
            position: fixed;
            top: 15px;
            right: 24px;
            width: 100px;
            z-index: 10020;
        }
        .st-key-home_nav_chat button,
        .st-key-home_nav_logout button {
            width: 100% !important;
            min-height: 34px !important;
            border-radius: 7px !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            border: 1px solid rgba(255,255,255,0.35) !important;
            background: transparent !important;
            color: #fff !important;
        }
        .st-key-home_nav_chat button:hover,
        .st-key-home_nav_logout button:hover {
            background: rgba(255,255,255,0.14) !important;
            color: #fff !important;
        }

        .stButton > button {
            background: #000 !important;
            color: #fff !important;
            border: 1px solid #000 !important;
        }

        .stButton > button:hover {
            background: #111 !important;
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="nav">
            <div class="logo">
                <div class="logo-dot"></div>
                <div class="logo-name">WELDPILOT AI</div>
            </div>
            <div class="nav-btns">
                <span class="btn-user">{username}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("채팅", key="home_nav_chat", use_container_width=True):
        on_start_chat()
        st.stop()

    if st.button("로그아웃", key="home_nav_logout", use_container_width=True):
        on_logout()
        st.stop()

    st.markdown("<div class='content-spacer'></div>", unsafe_allow_html=True)

    hero = st.container()
    with hero:
        title_col = st.columns([1, 2.5, 1])[1]
        with title_col:
            st.markdown(
                """
                <h1 style='text-align:center; font-size:48px; margin:0 0 18px 0; transform: translateX(14px);'>
                  데이터 기반 WELDPILOT과 함께하세요.
                </h1>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <p style='text-align:center; font-size:16px; line-height:1.6; margin:0 0 22px 0; font-family:"Noto Serif KR", serif;'>
                  WELDPILOT은 로봇용접 현장의 모든 위험을 감지하고 작업자를 지킵니다.
                  지금 바로 접속하여 안전한 작업 환경을 만드세요.
                </p>
                """,
                unsafe_allow_html=True,
            )

        btn_col = st.columns([1, 1, 1])[1]
        with btn_col:
            if st.button("지금 시작하세요.", use_container_width=True):
                on_start_chat()
                st.stop()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        img_col = st.columns([1, 2.2, 1])[1]
        with img_col:
            if IMAGE_PATH.exists():
                st.image(str(IMAGE_PATH), use_container_width=True)
            else:
                st.warning(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")
