import streamlit as st
import pandas as pd

from app.core.database import get_chat_logs


def show_admin_page():
    st.title("Admin Control Panel")
    st.markdown("WELD-BOT v4.0 관리자용 대시보드 및 데이터 관리 페이지")

    tab1, tab2, tab3 = st.tabs(["Chat Logs", "User Registrations", "PDF Ingestion"])

    with tab1:
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
            st.dataframe(df, use_container_width=True)
            st.subheader("User Activity Overview")
            user_counts = df["username"].value_counts()
            st.bar_chart(user_counts)

    with tab2:
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
                st.dataframe(user_df, use_container_width=True)
            else:
                st.info("가입한 사용자가 없습니다.")
        except Exception as e:
            st.error(f"사용자 정보를 불러오는 데 실패했습니다: {e}")

    with tab3:
        st.header("PDF Knowledge Base Uploader")
        st.markdown(
            """
            업로드한 PDF 문서를 다음 순서로 처리합니다:
            1. PDF 파싱 (marker / pypdf fallback)
            2. Markdown 변환 결과 로컬 저장
            3. 청킹(Chunking)
            4. 증분 임베딩 후 RDS PgVector 적재
            """
        )

        uploaded_file = st.file_uploader("Upload a PDF manual", type=["pdf"])

        if uploaded_file is not None:
            if st.button("Start Processing Pipeline", type="primary"):
                with st.spinner("Parsing PDF and running incremental embedding..."):
                    import requests

                    token = st.session_state.api_session.cookies.get("weld_auth_token")
                    if not token and "weld_auth_token" in st.context.cookies:
                        token = st.context.cookies["weld_auth_token"]

                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

                    try:
                        response = requests.post(
                            "http://localhost:8000/admin/upload_pdf",
                            files=files,
                            headers=headers,
                            timeout=600,
                        )

                        if response.status_code == 200:
                            data = response.json()
                            st.success(data["message"])
                            if data.get("parser_used"):
                                st.write(f"**Parser Used:** `{data['parser_used']}`")
                            if data.get("local_md_path"):
                                st.write(f"**Local MD Path:** `{data['local_md_path']}`")
                            st.write(f"**Total Chunks Parsed:** `{data.get('total_chunks_parsed', 0)}`")
                            st.write(f"**Vector Chunks Inserted:** `{data.get('db_chunks_inserted', 0)}`")
                            st.write(f"**Chunks Skipped (Already Indexed):** `{data.get('db_chunks_skipped', 0)}`")
                            st.write(f"**Stale Chunks Deleted:** `{data.get('db_chunks_deleted', 0)}`")
                            if data.get("file_hash"):
                                st.caption(f"file_hash: {data['file_hash']}")
                        else:
                            st.error(f"Pipeline Failed: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
