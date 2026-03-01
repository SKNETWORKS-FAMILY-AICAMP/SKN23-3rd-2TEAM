from pathlib import Path

import streamlit as st

IMAGE_PATH = Path(__file__).resolve().parent / "image" / "streamlit1.png"


def show_main_page(on_login, on_signup):
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
        [data-testid="stMain"] { padding-top: 0 !important; margin-top: 0 !important; }

        [data-testid="stMainBlockContainer"] { padding-top: 0 !important; margin-top: 0 !important; }

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
            transform: translateY(2px);
        }

        .btn-login {
            background: #000;
            color: #fff !important;
            border: 1.5px solid #000;
        }
        .btn-login:link,
        .btn-login:visited {
            color: #fff !important;
        }
        .btn-login:hover { background: #111; }

        .btn-signup {
            background: #fff;
            color: #000 !important;
            border: 1.5px solid rgba(0,0,0,0.15);
        }
        .btn-signup:hover { background: #f3f4f6; }

        .stButton > button {
            background: #000 !important;
            color: #fff !important;
            border: 1px solid #000 !important;
        }
        .stButton > button:hover {
            background: #111 !important;
            color: #fff !important;
        }

        .content-spacer { height: calc(var(--nav-h) + 10px); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="nav">
            <div class="logo">
                <a href="/?route=main" target="_self" style="text-decoration:none; display:flex; align-items:center; gap:10px;">
                    <div class="logo-dot"></div>
                    <div class="logo-name">WELDPILOT AI</div>
                </a>
            </div>
            <div class="nav-btns">
                <a class="btn-login" href="/?route=login" target="_self">로그인</a>
                <a class="btn-signup" href="/?route=signup" target="_self">회원가입</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                st.query_params["route"] = "login"
                st.rerun()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        img_col = st.columns([1, 2.2, 1])[1]
        with img_col:
            if IMAGE_PATH.exists():
                st.image(str(IMAGE_PATH), use_container_width=True)
            else:
                st.warning(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")
