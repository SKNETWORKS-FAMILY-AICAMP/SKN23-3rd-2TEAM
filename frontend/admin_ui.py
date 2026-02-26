import streamlit as st
import pandas as pd
from app.core.database import get_chat_logs

def show_admin_page():
    st.title("⚙️ Admin Control Panel")
    st.markdown("WELD-BOT v4.0 관리자 전용 대시보드 및 지식 베이스 관리 페이지입니다.")

    tab1, tab2, tab3 = st.tabs(["💬 Chat Logs", "👥 User Registrations", "📄 PDF Parser (Marker)"])

    # -------------------------------------------------------------
    # TAB 1: 실시간 채팅 로그 (Chat Logs)
    # -------------------------------------------------------------
    with tab1:
        st.header("Real-time Chat Interaction Logs")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Refresh Logs"):
                st.rerun()
        
        logs = get_chat_logs(limit=100)
        if not logs:
            st.info("기록된 대화 이력이 없습니다.")
        else:
            df = pd.DataFrame(logs)
            st.dataframe(df, use_container_width=True)
            
            st.subheader("User Activity Overview")
            user_counts = df["username"].value_counts()
            st.bar_chart(user_counts)

    # -------------------------------------------------------------
    # TAB 2: 회원가입 내역 (User Registrations)
    # -------------------------------------------------------------
    with tab2:
        st.header("User Registrations")
        st.write("최근 가입한 사용자 목록입니다.")
        
        # database.py 에 get_all_users() 함수가 있다고 가정하거나 직접 쿼리
        import psycopg2
        from app.core.database import get_connection_kwargs, open_optional_ssh_tunnel
        
        try:
            with open_optional_ssh_tunnel() as tunnel:
                conn_args = get_connection_kwargs()
                if tunnel:
                    conn_args["host"] = tunnel["host"]
                    conn_args["port"] = tunnel["port"]
                
                with psycopg2.connect(**conn_args) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC LIMIT 50")
                        users = cur.fetchall()
            if users:
                user_df = pd.DataFrame(users, columns=["ID", "Username", "Role", "Created At"])
                st.dataframe(user_df, use_container_width=True)
            else:
                st.info("가입된 사용자가 없습니다.")
        except Exception as e:
            st.error(f"사용자 정보를 불러오는 데 실패했습니다: {e}")

    # -------------------------------------------------------------
    # TAB 3: PDF 파일 업로드 및 AWS 파이프라인 연동 (Marker)
    # -------------------------------------------------------------
    with tab3:
        st.header("PDF Knowledge Base Uploader")
        st.markdown("""
        업로드한 PDF 문서는 다음과 같은 파이프라인을 거칩니다:
        1. **AWS S3** 원본 업로드
        2. **Marker** 모델을 통한 고품질 Markdown 변환
        3. 변환된 Markdown **AWS S3** 저장
        4. 청킹(Chunking)을 거쳐 **RDS PgVector** 벡터 DB에 적재
        """)

        uploaded_file = st.file_uploader("Upload a PDF manual", type=["pdf"])
        
        if uploaded_file is not None:
            if st.button("Start Processing Pipeline", type="primary"):
                with st.spinner("Processing PDF through Marker + AWS Pipeline..."):
                    import requests
                    
                    # session_state 에서 토큰 추출
                    token = st.session_state.api_session.cookies.get("weld_auth_token")
                    if not token and "weld_auth_token" in st.context.cookies:
                         token = st.context.cookies["weld_auth_token"]
                         
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    try:
                        # 백엔드의 /admin/upload_pdf API 호출
                        response = requests.post("http://localhost:8000/admin/upload_pdf", files=files, headers=headers)
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.success(data["message"])
                            st.write(f"📂 **S3 PDF Path:** `{data['s3_pdf_path']}`")
                            st.write(f"📄 **S3 MD Path:** `{data['s3_md_path']}`")
                            st.write(f"🧠 **Vector Chunks Inserted:** `{data['db_chunks_inserted']}`")
                        else:
                            st.error(f"Pipeline Failed: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")