def show_admin_page():
    import streamlit as st
    import pandas as pd
    import requests
    import psycopg2

    from app.core.database import (
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Ingestion",
        "🗂 Vector Registry",
        "👥 Users",
        "🤖 Model Settings",
        "🧹 Embedding 상태관리",
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
                with st.container(height=600):
                    st.code(st.session_state.preview_md, language="markdown")

            with col2:
                st.markdown("### Metadata JSON")
                with st.container(height=600):
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
    # TAB 3 - USERS
    # =====================================================
    with tab3:

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
                st.dataframe(user_df, width="stretch")
            else:
                st.info("가입 사용자 없음")

        except Exception as e:
            st.error(f"DB 오류: {e}")


    # =====================================================
    # TAB 4 - MODEL SETTINGS
    # =====================================================
    with tab4:
        st.markdown("## 🤖 LLM 모델 설정")
        st.caption("Fast/Accurate 모델을 관리자 페이지에서 즉시 변경할 수 있습니다.")

        token = st.session_state.get("access_token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        try:
            model_res = requests.get(
                f"{API_BASE}/admin/models",
                headers=headers,
                timeout=15,
            )
        except Exception as e:
            st.error(f"모델 설정 조회 실패: {e}")
            return

        if model_res.status_code != 200:
            st.error(model_res.text)
            return

        model_data = model_res.json()
        current_fast = model_data.get("model_fast", "")
        current_accurate = model_data.get("model_accurate", "")
        available_models = model_data.get("available_models", [])

        st.info(
            f"현재 설정 | Fast: `{current_fast}` | Accurate: `{current_accurate}`"
        )

        # 현재 사용 중 모델이 추천 목록에 없을 수도 있어 선택지에 항상 포함합니다.
        fast_options = list(dict.fromkeys(([current_fast] if current_fast else []) + available_models))
        accurate_options = list(dict.fromkeys(([current_accurate] if current_accurate else []) + available_models))

        if not fast_options:
            st.warning("사용 가능한 모델 목록이 비어 있습니다.")
            return

        with st.form("model_settings_form"):
            selected_fast = st.selectbox(
                "Fast 모델 (재작성/분류/검증)",
                fast_options,
                index=fast_options.index(current_fast) if current_fast in fast_options else 0,
            )
            selected_accurate = st.selectbox(
                "Accurate 모델 (최종 답변 생성)",
                accurate_options,
                index=accurate_options.index(current_accurate) if current_accurate in accurate_options else 0,
            )

            st.caption("목록에 없는 모델은 아래 커스텀 입력으로 지정할 수 있습니다.")
            custom_fast = st.text_input("Fast 모델 커스텀 입력 (선택)")
            custom_accurate = st.text_input("Accurate 모델 커스텀 입력 (선택)")

            save_btn = st.form_submit_button("💾 모델 설정 저장", type="primary")

        if save_btn:
            payload = {
                "model_fast": custom_fast.strip() or selected_fast,
                "model_accurate": custom_accurate.strip() or selected_accurate,
            }

            update_res = requests.put(
                f"{API_BASE}/admin/models",
                headers=headers,
                json=payload,
                timeout=15,
            )

            if update_res.status_code == 200:
                st.success("모델 설정이 저장되었습니다. 새 요청부터 즉시 반영됩니다.")
                st.rerun()

            error_detail = update_res.text
            try:
                error_detail = update_res.json().get("detail", error_detail)
            except Exception:
                pass
            st.error(f"저장 실패: {error_detail}")


    # =====================================================
    # TAB 5 - EMBEDDING STATUS MANAGEMENT
    # =====================================================
    with tab5:
        st.markdown("## 🧹 임베딩 비활성화 관리")
        st.caption("파일명/관리자/업로드일 검색 후 use_yn 상태를 변경할 수 있습니다.")

        token = st.session_state.get("access_token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if "embedding_search_rows" not in st.session_state:
            st.session_state.embedding_search_rows = []
        if "embedding_search_ran" not in st.session_state:
            st.session_state.embedding_search_ran = False

        with st.form("embedding_search_form"):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                search_file_name = st.text_input("파일명")
            with col2:
                search_creator = st.text_input("관리자 ID")
            with col3:
                search_date = st.text_input("업로드 날짜(YYYY-MM-DD)")
            with col4:
                search_use_yn = st.selectbox("상태", ["ALL", "Y", "N"], index=0)
            with col5:
                search_limit = st.selectbox("최대건수", [100, 200, 500, 1000], index=1)

            run_search = st.form_submit_button("🔍 검색", type="primary")

        if run_search:
            params = {"use_yn": search_use_yn, "limit": search_limit}
            if search_file_name.strip():
                params["file_name"] = search_file_name.strip()
            if search_creator.strip():
                params["creator"] = search_creator.strip()
            if search_date.strip():
                params["uploaded_date"] = search_date.strip()

            try:
                search_res = requests.get(
                    f"{API_BASE}/admin/embeddings/search",
                    headers=headers,
                    params=params,
                    timeout=20,
                )
                if search_res.status_code == 200:
                    st.session_state.embedding_search_rows = search_res.json().get("documents", [])
                    st.session_state.embedding_search_ran = True
                else:
                    st.error(search_res.text)
                    st.session_state.embedding_search_rows = []
                    st.session_state.embedding_search_ran = True
            except Exception as e:
                st.error(f"검색 실패: {e}")
                st.session_state.embedding_search_rows = []
                st.session_state.embedding_search_ran = True

        rows = st.session_state.get("embedding_search_rows", [])
        if not st.session_state.get("embedding_search_ran"):
            st.info("검색 조건을 입력하고 `검색` 버튼을 눌러주세요.")
        elif not rows:
            st.info("검색 결과가 없습니다.")
        else:
            df = pd.DataFrame(rows)
            view_col, pick_col = st.columns([3, 2])

            with view_col:
                st.dataframe(df, width="stretch")

            with pick_col:
                selector_df = df[["source_key", "creator", "use_yn"]].copy()
                selector_df.insert(0, "선택", False)
                edited_selector_df = st.data_editor(
                    selector_df,
                    hide_index=True,
                    width="stretch",
                    disabled=["source_key", "creator", "use_yn"],
                    column_config={
                        "선택": st.column_config.CheckboxColumn("선택"),
                        "source_key": st.column_config.TextColumn("파일"),
                        "creator": st.column_config.TextColumn("관리자"),
                        "use_yn": st.column_config.TextColumn("상태"),
                    },
                    key="embedding_selector_editor",
                )
                selected_sources = edited_selector_df.loc[
                    edited_selector_df["선택"], "source_key"
                ].tolist()
                st.caption(f"선택된 파일: {len(selected_sources)}개")

            col_a, col_b = st.columns(2)
            with col_a:
                deactivate_btn = st.button("⛔ 선택 비활성화 (N)", type="primary")
            with col_b:
                activate_btn = st.button("✅ 선택 활성화 (Y)")

            if deactivate_btn and selected_sources:
                res = requests.put(
                    f"{API_BASE}/admin/embeddings/use_yn",
                    headers=headers,
                    json={"source_keys": selected_sources, "use_yn": "N"},
                    timeout=30,
                )
                if res.status_code == 200:
                    st.success("비활성화 완료 (BM25 캐시에서는 해당 문서가 제거됨)")
                    st.rerun()
                else:
                    st.error(res.text)
            elif deactivate_btn:
                st.warning("비활성화할 파일을 먼저 체크해 주세요.")

            if activate_btn and selected_sources:
                res = requests.put(
                    f"{API_BASE}/admin/embeddings/use_yn",
                    headers=headers,
                    json={"source_keys": selected_sources, "use_yn": "Y"},
                    timeout=30,
                )
                if res.status_code == 200:
                    st.success("활성화 완료")
                    st.rerun()
                else:
                    st.error(res.text)
            elif activate_btn:
                st.warning("활성화할 파일을 먼저 체크해 주세요.")


