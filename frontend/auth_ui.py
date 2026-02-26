import streamlit as st

def show_auth_page(cookie_manager, api, API_URL):
    """인증 페이지 UI 및 쿠키 기반 세션 복구 로직 (Anti-Flicker 적용)"""
    st.title("🔥 WELD·BOT v4.0 Access Control")
    st.caption("FastAPI Backend + Browser Cookie Persistence (Optimized)")
    
    # OAuth Callback 처리
    auth_code = st.query_params.get("auth_code")
    if auth_code and not st.session_state.get("authenticated"):
        try:
            response = api.post(f"{API_URL}/auth/oauth/exchange", data={"code": auth_code}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.user = data["user"]
                st.session_state.authenticated = True
                token = data.get("weld_auth_token")
                if token:
                    cookie_manager.set("weld_auth_token", token)
                st.query_params.clear()
                st.rerun()
            else:
                st.error("소셜 로그인 인증에 실패했습니다.")
        except Exception as e:
            st.error(f"소셜 로그인 연결 오류: {e}")

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
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.user = data["user"]
                    st.session_state.authenticated = True
                    
                    # 🚨 핵심: 브라우저 쿠키에 토큰 저장 (F5 방어)
                    token = response.cookies.get("weld_auth_token")
                    if token:
                        cookie_manager.set("weld_auth_token", token)
                    
                    st.success(f"{login_id}님, 환영합니다!")
                    st.rerun()
                else:
                    st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
            except Exception as e:
                st.error(f"백엔드 연결 실패: {e}")
                
        st.markdown("---")
        st.markdown("### 소셜 로그인")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f'<a href="{API_URL}/auth/oauth/google/login?frontend_redirect_uri=http://localhost:8501" target="_self"><button style="width:100%">Google Login</button></a>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<a href="{API_URL}/auth/oauth/kakao/login?frontend_redirect_uri=http://localhost:8501" target="_self"><button style="width:100%">Kakao Login</button></a>', unsafe_allow_html=True)
                
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
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.success("회원가입이 완료되었습니다! 로그인 탭에서 로그인해 주세요.")
                    else:
                        st.error("이미 존재하는 아이디이거나 회원가입 중 오류가 발생했습니다.")
                except Exception as e:
                    st.error(f"백엔드 연결 실패: {e}")

def logout(cookie_manager, api, API_URL):
    """로그아웃 및 쿠키 삭제"""
    try:
        api.post(f"{API_URL}/auth/logout", timeout=5)
    except:
        pass
    
    # 쿠키 및 세션 초기화
    cookie_manager.delete("weld_auth_token")
    st.session_state.user = None
    st.session_state.authenticated = False
    st.session_state.cookie_mounted = False
    
    import requests
    st.session_state.api_session = requests.Session()
    st.rerun()
