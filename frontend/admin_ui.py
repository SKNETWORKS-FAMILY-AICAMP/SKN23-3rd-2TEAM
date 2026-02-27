import streamlit as st
import pandas as pd

from app.core.database import get_chat_logs


def show_admin_page():
    st.title("Admin Control Panel")
    st.markdown("WELD-BOT v4.0 관리자용 대시보드 및 데이터 관리 페이지")

    tab1, tab2, tab3, tab4 = st.tabs([
        "PDF 파싱 & 적재 (Ingestion)", 
        "Vector DB 관리 (Search & Delete)", 
        "Chat Logs", 
        "User Registrations"
    ])

    with tab1:
        st.header("Step 1 & 2: PDF Parsing and S3/PGVector Ingestion")
        
        admin_name = st.text_input("승인 관리자 성명 (사번/이름) *필수", key="admin_name", placeholder="예: 12345/홍길동")
        
        parser_choice = st.radio(
            "파싱 엔진 선택",
            ["marker", "pymupdf4llm"],
            captions=["기본 (품질 우수, 형태 유지 좋음)", "한글 깨짐 방지 대안 파서 (RAG 친화적 마크다운)"],
            horizontal=True
        )

        uploaded_file = st.file_uploader("Upload a PDF manual", type=["pdf"], key="pdf_uploader")
        
        # Initialize session state for preview
        if "preview_md" not in st.session_state:
            st.session_state.preview_md = None
        if "preview_json" not in st.session_state:
            st.session_state.preview_json = None
        if "preview_filename" not in st.session_state:
            st.session_state.preview_filename = None
        if "preview_filebytes" not in st.session_state:
            st.session_state.preview_filebytes = None
        if "already_exists" not in st.session_state:
            st.session_state.already_exists = False

        if uploaded_file is not None and admin_name.strip():
            # Only allow parsing if admin_name is filled and file is uploaded
            if st.button("파싱 시작 (미리보기)", type="secondary"):
                with st.spinner("파싱 진행 중 (DB 적재 안함)..."):
                    import requests
                    token = st.session_state.get("access_token")
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    file_bytes = uploaded_file.getvalue()
                    files = {"file": (uploaded_file.name, file_bytes, "application/pdf")}
                    data_payload = {
                        "parser": parser_choice,
                        "admin_name": admin_name
                    }
                    
                    try:
                        response = requests.post(
                            "http://localhost:8000/admin/parse_pdf_preview",
                            files=files,
                            data=data_payload,
                            headers=headers,

                            timeout=600,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.preview_md = data.get("markdown_text")
                            st.session_state.preview_json = data.get("metadata_json")
                            st.session_state.preview_filename = uploaded_file.name
                            st.session_state.preview_filebytes = file_bytes
                            st.session_state.already_exists = data.get("already_exists", False)
                            st.success(data["message"])
                        else:
                            st.error(f"Parsing Failed: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
        elif uploaded_file is not None and not admin_name.strip():
            st.warning("먼저 승인 관리자 성명을 입력해주세요.")

        # If preview data is in session, show Step 2 (Preview & Commit)
        if st.session_state.preview_md and st.session_state.preview_json:
            st.markdown("---")
            st.subheader("Step 2: 검증 및 승인")
            
            if st.session_state.already_exists:
                st.warning("⚠️ **이미 동일한 파일명으로 등록된 문서가 DB에 존재합니다!** '승인 및 DB 적재' 시 기존 데이터가 덮어씌워(Update) 집니다.")
                
            col_md, col_json = st.columns(2)
            with col_md:
                st.markdown("**Markdown 본문 미리보기**")
                st.text_area("Markdown Content", st.session_state.preview_md, height=400, disabled=True)
            with col_json:
                st.markdown("**메타데이터 JSON 미리보기**")
                st.text_area("Metadata JSON", st.session_state.preview_json, height=400, disabled=True)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("✅ 승인 및 DB 적재", type="primary"):
                    with st.spinner("S3 업로드 및 Vector DB 적재 중..."):
                        import requests
                        token = st.session_state.get("access_token")
                        headers = {"Authorization": f"Bearer {token}"} if token else {}
                        
                        payload = {
                            "markdown_text": st.session_state.preview_md,
                            "metadata_json": st.session_state.preview_json,
                            "admin_name": admin_name
                        }
                        
                        files = {
                            "file": (st.session_state.preview_filename, st.session_state.preview_filebytes, "application/pdf")
                        }
                        
                        try:
                            response = requests.post(
                                "http://localhost:8000/admin/commit_pdf",
                                files=files,
                                data=payload,
                                headers=headers,
                                timeout=600,
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.success("✅ DB 반영 및 S3 처리가 완료되었습니다! 검색 인덱스(BM25)는 백그라운드에서 최적화 중입니다.")
                                st.write(f"**Total Chunks Parsed:** `{data.get('total_chunks_parsed', 0)}`")
                                st.write(f"**Vector Chunks Inserted:** `{data.get('db_chunks_inserted', 0)}`")
                                
                                # Clear state
                                st.session_state.preview_md = None
                                st.session_state.preview_json = None
                                st.session_state.preview_filename = None
                                st.session_state.preview_filebytes = None
                                st.session_state.already_exists = False
                            else:
                                st.error(f"Commit Failed: {response.text}")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")
            with col_btn2:
                if st.button("❌ 반려 및 취소"):
                    st.session_state.preview_md = None
                    st.session_state.preview_json = None
                    st.session_state.preview_filename = None
                    st.session_state.preview_filebytes = None
                    st.session_state.already_exists = False
                    st.rerun()

    with tab2:
        st.header("Vector DB 데이터 관리 (Search & Delete)")
        
        # Load registry data upon opening tab
        if st.button("목록 새로고침", key="refresh_registry"):
            pass
            
        import requests
        token = st.session_state.get("access_token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        
        try:
            response = requests.get("http://localhost:8000/admin/registry", headers=headers)
            if response.status_code == 200:
                docs = response.json().get("documents", [])
                if docs:
                    df = pd.DataFrame(docs)
                    df.rename(columns={
                        "source_key": "문서명",
                        "chunk_count": "청크 갯수", 
                        "created_at": "업로드 일시",
                        "creator": "관리자",
                        "s3_pdf_key": "S3 PDF 경로",
                        "s3_md_key": "S3 MD 경로",
                    }, inplace=True)
                    
                    # Selection form
                    st.dataframe(df, width="stretch")
                    st.markdown("---")
                    st.subheader("선택한 데이터 삭제")
                    delete_options = st.multiselect("삭제할 문서(source_key)를 선택하세요", options=df["문서명"].tolist())
                    
                    if st.button("🚨 선택한 데이터 DB/S3에서 모두 삭제", type="primary", disabled=len(delete_options)==0):
                        with st.spinner("Deleting from DB and S3..."):
                            del_response = requests.delete(
                                "http://localhost:8000/admin/registry",
                                headers=headers,
                                json={"source_keys": delete_options}
                            )
                            if del_response.status_code == 200:
                                st.success("✅ DB 반영 및 S3 처리가 완료되었습니다! 검색 인덱스(BM25)는 백그라운드에서 최적화 중입니다.")
                                st.rerun()
                            else:
                                st.error(f"Delete Failed: {del_response.text}")
                else:
                    st.info("현재 파싱되어 저장된 문서가 없습니다.")
            else:
                st.error(f"목록 조회 실패: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

    with tab3:
        st.header("Real-time Chat Interaction Logs")
        col1, _col2 = st.columns([1, 4])
        with col1:
            if st.button("Refresh Logs"):
                st.rerun()

        logs = get_chat_logs(limit=100)
        if not logs:
            st.info("기록된 대화 로그가 없습니다.")
        else:
            df = pd.DataFrame(logs)
            st.dataframe(df, width="stretch")
            st.subheader("User Activity Overview")
            user_counts = df["username"].value_counts()
            st.bar_chart(user_counts)

    with tab4:
        st.header("User Registrations")
        st.write("최근 가입한 사용자 목록입니다.")

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
                        cur.execute(
                            "SELECT id, username, role, created_at FROM users ORDER BY created_at DESC LIMIT 50"
                        )
                        users = cur.fetchall()
            if users:
                user_df = pd.DataFrame(users, columns=["ID", "Username", "Role", "Created At"])
                st.dataframe(user_df, width="stretch")
            else:
                st.info("가입한 사용자가 없습니다.")
        except Exception as e:
            st.error(f"사용자 정보를 불러오는 데 실패했습니다: {e}")
