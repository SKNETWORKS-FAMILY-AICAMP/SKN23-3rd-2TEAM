from pathlib import Path

import streamlit as st

IMAGE_DIR = Path(__file__).resolve().parent / "image"


def _resolve_image_path() -> Path:
    for file_name in ("streamlit1.png", "streamlit1.jpg"):
        candidate = IMAGE_DIR / file_name
        if candidate.exists():
            return candidate
    return IMAGE_DIR / "streamlit1.png"


IMAGE_PATH = _resolve_image_path()


def show_main_page_logout(on_logout, on_start_chat):
    if st.query_params.get("action") == "logout":
        st.query_params.clear()
        on_logout()
        return

    username = st.session_state.get("user_name", "사용자")

    # -----------------------------
    # Navigation & Unified Style
    # -----------------------------

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;500;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&display=swap');

        :root {{
            --nav-h: 64px;
            --line: rgba(120, 190, 220, 0.20);
            --primary: #ff6a3d;
            --secondary: #2ec5ff;
        }}

        html, body {{ margin: 0 !important; padding: 0 !important; }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none !important; }}
        .stApp {{ background-color: #f2f2f2; }}
        [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
            padding-top: 0 !important; margin-top: 0 !important;
        }}

        .nav {{
            display: flex; align-items: center; justify-content: space-between;
            height: var(--nav-h); position: fixed;
            top: 0; left: 0; right: 0;
            padding: 0 32px 0 24px;
            background: #000 !important; border-bottom: 1px solid var(--line); z-index: 9998;
        }}

        .logo {{ display: flex; align-items: center; gap: 10px; font-family: 'Chakra Petch', sans-serif; text-decoration: none !important; color: inherit !important; }}
        .logo-dot {{
            width: 14px; height: 14px; border-radius: 50%;
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            box-shadow: 0 0 20px rgba(46,197,255,0.8);
        }}
        .logo-name {{ font-size: 0.95rem; font-weight: 600; color: #dbf5ff !important; letter-spacing: 0.08em; }}

        /* Status & Header Spacing */
        .chat-online-status-header {{
            position: fixed !important; 
            top: 22px !important; 
            right: 160px !important;
            color: #a7f3d0 !important; 
            font-size: 0.83rem !important; 
            font-weight: 700 !important;
            text-shadow: 0 0 10px rgba(34,197,94,0.28);
            white-space: nowrap; 
            z-index: 10000 !important;
        }}

        .content-spacer {{ height: calc(var(--nav-h) + 60px); }}

        /* Navigation Buttons */
        .st-key-home_nav_logout {{ position: fixed; top: 15px; right: 24px; width: 100px; z-index: 10020; }}

        .st-key-home_nav_logout button {{
            width: 100% !important; min-height: 34px !important;
            border-radius: 7px !important; font-size: 0.78rem !important;
            font-weight: 700 !important; border: 1.5px solid var(--primary) !important;
            background: transparent !important; color: var(--primary) !important;
            transition: all 0.2s;
        }}

        .st-key-home_nav_logout button:hover {{
            background: rgba(255,106,61,0.12) !important;
            box-shadow: 0 0 12px rgba(255,106,61,0.2);
        }}

        /* Start Button Custom Styling */
        div.stButton > button[kind="secondary"] {{
            /* Fallback selector if key fails */
        }}
        
        .st-key-home_start_btn button {{
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
            font-weight: 700 !important;
            height: 48px !important;
            font-size: 1.1rem !important;
            border-radius: 8px !important;
            transition: all 0.3s !important;
        }}
        .st-key-home_start_btn button:hover {{
            background-color: #333333 !important;
            border-color: #333333 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        /* Utility to hide legacy elements */
        .chat-online-status, .st-key-home_nav_chat, .st-key-home_nav_admin {{ display: none !important; }}
        </style>

        <div class="nav">
            <a class="logo" href="/?route=home" target="_self">
                <div class="logo-dot"></div>
                <div class="logo-name">WELDPILOT AI</div>
            </a>
            <div class="chat-online-status-header">
                🟢 {username}님 접속 중
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Logout", key="home_nav_logout"):
        st.query_params["route"] = "logout"
        st.rerun()

    # CSS adjustment: hide the other nav keys if rendered by mistake
    st.markdown("<style>.st-key-home_nav_chat, .st-key-home_nav_admin, .chat-online-status { display:none !important; }</style>", unsafe_allow_html=True)

    st.markdown("<div class='content-spacer'></div>", unsafe_allow_html=True)

    hero = st.container()
    with hero:
        title_col = st.columns([1, 2.5, 1])[1]
        with title_col:
            st.markdown(
                """
                <h1 style='text-align:center; font-size:48px; margin:0 0 18px 0;'>
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

        btn_container = st.columns([1, 1, 1])[1]
        with btn_container:
            if st.button("지금 시작하세요.", key="home_start_btn", use_container_width=True):
                on_start_chat()
                st.stop()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        img_col = st.columns([1, 2.2, 1])[1]
        with img_col:
            if IMAGE_PATH.exists():
                st.image(str(IMAGE_PATH), use_container_width=True)
            else:
                st.warning(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")
