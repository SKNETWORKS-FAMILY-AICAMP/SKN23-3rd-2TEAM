def show_admin_page():
    import streamlit as st
    import pandas as pd
    import requests
    import psycopg2

    from app.core.database import (
        get_chat_logs,
        get_connection_kwargs,
        open_optional_ssh_tunnel
    )

    API_BASE = "http://localhost:8000"

    # -----------------------------
    # Page Config
    # -----------------------------
    st.set_page_config(
        page_title="WELD-BOT Admin",
        page_icon="🤖",
        layout="wide"
    )

    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.title("🤖 WELD-BOT v4.0")
        st.markdown("### 🔐 Admin Panel")

        st.write(f"접속 관리자: `{st.session_state.get('username', 'Unknown')}`")

        st.divider()
        st.markdown("### 📡 System Status")
        st.success("API Server: Online")
        st.success("Vector DB: Connected")

        st.divider()
        if st.button("🔄 새로고침"):
            st.rerun()


    # -----------------------------
    # Main Title
    # -----------------------------
    st.title("📊 관리자 통합 대시보드")
    st.caption("WELD-BOT 문서 관리 · Vector DB · 로그 모니터링")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 Ingestion",
        "🗂 Vector Registry",
        "💬 Chat Logs",
        "👥 Users"
    ])

    # =====================================================
    # TAB 1 - INGESTION
    # =====================================================
    with tab1:

        st.markdown("## 📥 PDF Ingestion Pipeline")
        st.info("1️⃣ Parse → 2️⃣ Preview → 3️⃣ Commit")

        admin_name = st.text_input(
            "승인 관리자 성명 (사번/이름)",
            placeholder="예: 12345/홍길동"
        )

        parser_choice = st.radio(
            "파싱 엔진 선택",
            ["marker", "pymupdf4llm"],
            horizontal=True
        )

        uploaded_file = st.file_uploader(
            "PDF 매뉴얼 업로드",
            type=["pdf"]
        )

        if "preview_md" not in st.session_state:
            st.session_state.preview_md = None
            st.session_state.preview_json = None
            st.session_state.preview_filename = None
            st.session_state.preview_filebytes = None
            st.session_state.already_exists = False

        if uploaded_file and admin_name:

            if st.button("🚀 파싱 시작 (Preview)"):
                with st.spinner("Parsing..."):
                    token = st.session_state.get("access_token")
                    headers = {"Authorization": f"Bearer {token}"} if token else {}

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }

                    data_payload = {
                        "parser": parser_choice,
                        "admin_name": admin_name
                    }

                    res = requests.post(
                        f"{API_BASE}/admin/parse_pdf_preview",
                        files=files,
                        data=data_payload,
                        headers=headers,
                        timeout=600
                    )

                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.preview_md = data["markdown_text"]
                        st.session_state.preview_json = data["metadata_json"]
                        st.session_state.preview_filename = uploaded_file.name
                        st.session_state.preview_filebytes = uploaded_file.getvalue()
                        st.session_state.already_exists = data.get("already_exists", False)
                        st.success("파싱 완료")
                    else:
                        st.error(res.text)

        # Preview Section
        if st.session_state.preview_md:

            st.divider()
            st.subheader("🔎 Preview")

            if st.session_state.already_exists:
                st.warning("⚠️ 동일 파일 존재 → 승인 시 기존 데이터 덮어쓰기")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Markdown")
                st.code(st.session_state.preview_md, language="markdown")

            with col2:
                st.markdown("### Metadata JSON")
                st.code(st.session_state.preview_json, language="json")

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("✅ 승인 및 DB 적재", type="primary"):
                    with st.spinner("Committing..."):
                        token = st.session_state.get("access_token")
                        headers = {"Authorization": f"Bearer {token}"} if token else {}

                        files = {
                            "file": (
                                st.session_state.preview_filename,
                                st.session_state.preview_filebytes,
                                "application/pdf"
                            )
                        }

                        payload = {
                            "markdown_text": st.session_state.preview_md,
                            "metadata_json": st.session_state.preview_json,
                            "admin_name": admin_name
                        }

                        res = requests.post(
                            f"{API_BASE}/admin/commit_pdf",
                            files=files,
                            data=payload,
                            headers=headers,
                            timeout=600
                        )

                        if res.status_code == 200:
                            data = res.json()
                            st.success("DB 반영 완료")
                            st.metric("Total Chunks", data.get("total_chunks_parsed", 0))
                            st.metric("Inserted", data.get("db_chunks_inserted", 0))
                            st.session_state.preview_md = None
                            st.rerun()
                        else:
                            st.error(res.text)

            with col_btn2:
                if st.button("❌ 취소"):
                    st.session_state.preview_md = None
                    st.rerun()


    # =====================================================
    # TAB 2 - VECTOR REGISTRY
    # =====================================================
    with tab2:

        st.markdown("## 🗂 Vector DB Registry")

        token = st.session_state.get("access_token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        res = requests.get(f"{API_BASE}/admin/registry", headers=headers)

        if res.status_code == 200:
            docs = res.json().get("documents", [])

            if docs:
                df = pd.DataFrame(docs)

                col1, col2, col3 = st.columns(3)
                col1.metric("총 문서 수", len(df))
                col2.metric("총 청크 수", df["chunk_count"].sum())
                col3.metric("최근 업로드", df["created_at"].max())

                st.dataframe(df, use_container_width=True)

                st.warning("⚠️ 삭제 시 DB + S3 + 인덱스 모두 제거됩니다")

                delete_targets = st.multiselect(
                    "삭제할 문서 선택",
                    df["source_key"].tolist()
                )

                if st.button("🚨 선택 삭제", type="primary") and delete_targets:
                    del_res = requests.delete(
                        f"{API_BASE}/admin/registry",
                        headers=headers,
                        json={"source_keys": delete_targets}
                    )
                    if del_res.status_code == 200:
                        st.success("삭제 완료")
                        st.rerun()
                    else:
                        st.error(del_res.text)
            else:
                st.info("저장된 문서 없음")


    # =====================================================
    # TAB 3 - CHAT LOGS
    # =====================================================
    with tab3:

        st.markdown("## 💬 Chat Interaction Logs")

        logs = get_chat_logs(limit=200)

        if logs:
            df = pd.DataFrame(logs)

            user_filter = st.selectbox(
                "사용자 필터",
                ["All"] + list(df["username"].unique())
            )

            if user_filter != "All":
                df = df[df["username"] == user_filter]

            search_query = st.text_input("질문 검색")

            if search_query:
                df = df[df["question"].str.contains(search_query, case=False)]

            st.dataframe(df, use_container_width=True)

            st.subheader("📈 User Activity")
            st.bar_chart(df["username"].value_counts())
        else:
            st.info("로그 없음")


    # =====================================================
    # TAB 4 - USERS
    # =====================================================
    with tab4:

        st.markdown("## 👥 User Registrations")

        try:
            with open_optional_ssh_tunnel() as tunnel:
                conn_args = get_connection_kwargs()

                if tunnel:
                    conn_args["host"] = tunnel["host"]
                    conn_args["port"] = tunnel["port"]

                with psycopg2.connect(**conn_args) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT id, username, role, created_at
                            FROM users
                            ORDER BY created_at DESC
                            LIMIT 50
                        """)
                        users = cur.fetchall()

            if users:
                user_df = pd.DataFrame(
                    users,
                    columns=["ID", "Username", "Role", "Created At"]
                )
                st.dataframe(user_df, use_container_width=True)
            else:
                st.info("가입 사용자 없음")

        except Exception as e:
            st.error(f"DB 오류: {e}")