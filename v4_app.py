import streamlit as st
import time
import requests
import extra_streamlit_components as stx

# 1. Page Config
st.set_page_config(
    page_title="WELD·BOT v4.0",
    page_icon="🔥",
    layout="wide"
)

# 1. 쿠키 매니저 초기화 (캐싱 데코레이터 제거)
# 컴포넌트를 직접 호출하여 Streamlit의 위젯 캐싱 경고를 우회합니다.
cookie_manager = stx.CookieManager(key="auth_cookie_manager")

API_URL = "http://localhost:8000"

# 3. Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "api_session" not in st.session_state:
    st.session_state.api_session = requests.Session()
if "cookie_mounted" not in st.session_state:
    st.session_state.cookie_mounted = False

api = st.session_state.api_session

# 👉 stx.CookieManager 대신 Streamlit 네이티브 기능으로 쿠키 즉시 로드 (새로고침 깜빡임 원천 차단)
# 4. JWT 쿠키 검증 및 세션 복구 로직
if not st.session_state.authenticated:
    # 캐시 지연이나 JS Iframe 렌더링 딜레이 없이 0초 만에 쿠키 조회
    saved_token = st.context.cookies.get("weld_auth_token")
    if saved_token:
        try:
            response = api.get(f"{API_URL}/auth/me", headers={"Authorization": f"Bearer {saved_token}"}, timeout=5)
            if response.status_code == 200:
                st.session_state.user = response.json()
                st.session_state.authenticated = True
                st.rerun()
            else:
                cookie_manager.delete("weld_auth_token")
        except:
            pass

# 5. Routing
if not st.session_state.authenticated:
    # 비로그인 상태 -> Auth UI
    from frontend.auth_ui import show_auth_page
    show_auth_page(cookie_manager, api, API_URL)
else:
    # 로그인 상태: 사이드바 구성 -> Chat / Admin UI
    from frontend.auth_ui import logout
    from frontend.chat_ui import show_chat_page
    from frontend.admin_ui import show_admin_page

    user = st.session_state.user
    with st.sidebar:
        st.title("🔥 WELD·BOT v4.0")
        st.write(f"Logged in as: **{user['username']}** ({user['role']})")
        
        # 권한에 따른 메뉴 필터링
        menu_options = ["💬 Chatbot"]
        if user["role"] == "admin":
            menu_options.append("⚙️ Admin Dashboard")
        
        selection = st.radio("Navigation", menu_options)
        
        st.divider()
        if st.button("Logout", use_container_width=True):
            logout(cookie_manager, api, API_URL)

    # 6. 화면 라우팅
    if selection == "💬 Chatbot":
        show_chat_page()
    elif selection == "⚙️ Admin Dashboard" and user["role"] == "admin":
        show_admin_page()
