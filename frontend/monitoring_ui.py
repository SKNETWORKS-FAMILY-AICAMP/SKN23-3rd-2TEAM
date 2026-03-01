import pandas as pd
import streamlit as st

from app.core.database import get_chat_logs

def show_monitoring_page():
    # -----------------------------
    # Navigation & Style
    # -----------------------------
    username = st.session_state.get("user_name", "관리자")
    role = st.session_state.get("user", {}).get("role", "admin")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;500;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&display=swap');

        :root {
            --nav-h: 64px;
            --line: rgba(120, 190, 220, 0.20);
            --primary: #ff6a3d;
            --secondary: #2ec5ff;
        }

        html, body { margin: 0 !important; padding: 0 !important; }
        [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
        .stApp { background-color: #f2f2f2; }
        [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
            padding-top: 0 !important; margin-top: 0 !important;
        }

        .nav {
            display: flex; align-items: center; justify-content: space-between;
            height: var(--nav-h); position: fixed;
            top: 0; left: 0; right: 0;
            padding: 0 32px 0 24px;
            background: #000; border-bottom: 1px solid var(--line); z-index: 9998;
        }

        .logo { display: flex; align-items: center; gap: 10px; font-family: 'Chakra Petch', sans-serif; text-decoration: none !important; }
        .logo-dot {
            width: 14px; height: 14px; border-radius: 50%;
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            box-shadow: 0 0 20px rgba(46,197,255,0.8);
        }
        .logo-name { font-size: 0.95rem; font-weight: 600; color: #dbf5ff; letter-spacing: 0.08em; }

        .chat-online-status {
            position: fixed; top: 21px; right: 410px; z-index: 10025;
            color: #a7f3d0; font-size: 0.83rem; font-weight: 700;
            letter-spacing: 0.01em; pointer-events: none;
            text-shadow: 0 0 10px rgba(34,197,94,0.28);
            white-space: nowrap;
        }

        .content-spacer { height: calc(var(--nav-h) + 20px); }

        /* Navigation Buttons */
        .st-key-mon_nav_home { position: fixed; top: 15px; right: 300px; width: 80px; z-index: 10020; }
        .st-key-mon_nav_chat { position: fixed; top: 15px; right: 212px; width: 80px; z-index: 10020; }
        .st-key-mon_nav_admin { position: fixed; top: 15px; right: 124px; width: 80px; z-index: 10020; }
        .st-key-mon_nav_logout { position: fixed; top: 15px; right: 24px; width: 90px; z-index: 10020; }

        .st-key-mon_nav_home button, .st-key-mon_nav_chat button, .st-key-mon_nav_admin button, .st-key-mon_nav_logout button {
            width: 100% !important; min-height: 34px !important;
            border-radius: 7px !important; font-size: 0.78rem !important;
            font-weight: 700 !important; border: 1px solid transparent !important;
            background: transparent !important;
        }

        .st-key-mon_nav_home button { border-color: #fff !important; color: #fff !important; }
        .st-key-mon_nav_home button:hover { background: rgba(255,255,255,0.1) !important; }

        .st-key-mon_nav_chat button { border-color: var(--secondary) !important; color: var(--secondary) !important; }
        .st-key-mon_nav_chat button:hover { background: rgba(46,197,255,0.1) !important; }
        
        .st-key-mon_nav_admin button { border-color: #22c55e !important; color: #22c55e !important; }
        .st-key-mon_nav_admin button:hover { background: rgba(34,197,94,0.1) !important; }

        .st-key-mon_nav_logout button { border-color: var(--primary) !important; color: var(--primary) !important; }
        .st-key-mon_nav_logout button:hover { background: rgba(255,106,61,0.1) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Standard Nav Bar
    st.markdown(
        f"""
        <div class="nav">
            <a class="logo" href="/?route=home" target="_self">
                <div class="logo-dot"></div>
                <div class="logo-name">WELDPILOT AI</div>
            </a>
            <div class="header-right">
                <span class="chat-online-status-header-mon">🟢 {username}님 접속 중</span>
            </div>
        </div>
        <style>
        .header-right {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-left: auto;
            position: absolute;
            right: 480px; /* Offset for the 4 buttons */
            top: 50%;
            transform: translateY(-50%);
        }}
        .chat-online-status-header-mon {{
            color: #a7f3d0;
            font-size: 0.83rem;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(34,197,94,0.28);
            white-space: nowrap;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render Navigation Buttons
    if st.button("Home", key="mon_nav_home"):
        st.query_params["route"] = "home"
        st.session_state.auth_route = "home"
        st.rerun()
    if st.button("Chat", key="mon_nav_chat"):
        st.query_params["route"] = "chat"
        st.session_state.auth_route = "chat"
        st.rerun()
    if role == "admin":
        if st.button("Admin", key="mon_nav_admin"):
            st.query_params["route"] = "settings"
            st.session_state.auth_route = "settings"
            st.rerun()
    if st.button("Logout", key="mon_nav_logout"):
        st.query_params["route"] = "logout"
        st.rerun()

    st.markdown("<div class='content-spacer'></div>", unsafe_allow_html=True)
    st.title("📊 RAG 품질 모니터링")
    st.caption("AI 심판(LLM-as-a-Judge)이 측정한 시스템 건강 상태 및 오답 감점 사유 로그입니다.")

    try:
        logs = get_chat_logs(limit=1000)
    except Exception as e:
        st.error(f"DB Error fetching logs: {e}")
        logs = []

    if logs:
        df = pd.DataFrame(logs)
        if "latency" in df.columns:
            df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
        if "generation_score" in df.columns:
            df["generation_score"] = pd.to_numeric(df["generation_score"], errors="coerce")
            
        # --- 영역 A: 상단 요약 통계 ---
        st.subheader("📈 시스템 헬스 체크 지표")
        if "generation_score" in df.columns:
            valid_gen = df.dropna(subset=["generation_score"])
            gen_pass_rate = valid_gen["generation_score"].mean() * 100 if not valid_gen.empty else 0
            
            valid_ret = df.dropna(subset=["retrieval_total_chunks"])
            if not valid_ret.empty:
                ret_rel_rate = (valid_ret["retrieval_relevant_chunks"] / valid_ret["retrieval_total_chunks"].replace(0, 1)).mean() * 100
            else:
                ret_rel_rate = 0
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("📊 답변 충실도 (Pass Rate)", f"{gen_pass_rate:.1f}%")
            col_m2.metric("📊 문서 검색 품질 (F1 Score)", f"{ret_rel_rate:.1f}%")
            
        st.divider()

        # --- 영역 B: 하단 User Activity 전면 개편 ---
        st.subheader("📈 시스템 이용 현황")
        col_act1, col_act2 = st.columns([3, 2])
        
        with col_act1:
            st.markdown("**일별 질문 추이**")
            if "timestamp" in df.columns and not df.empty:
                try:
                    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                    daily_counts = df.groupby('date').size().reset_index(name='count')
                    if not daily_counts.empty:
                        st.line_chart(daily_counts.set_index('date'), y='count')
                    else:
                        st.info("데이터가 충분하지 않습니다.")
                except Exception as e:
                    st.error(f"그래프 렌더링 오류: {e}")
            else:
                st.info("데이터가 충분하지 않습니다.")

        with col_act2:
            st.markdown("**Top 사용자 현황**")
            if "username" in df.columns and "latency" in df.columns:
                user_stats = df.groupby('username').agg(
                    질문건수=('id', 'count'),
                    평균응답속도=('latency', 'mean')
                ).reset_index()
                user_stats = user_stats.sort_values(by='질문건수', ascending=False).head(10)
                user_stats['평균응답속도(초)'] = user_stats['평균응답속도'].round(2)
                user_stats = user_stats.drop(columns=['평균응답속도'])
                user_stats = user_stats.rename(columns={'username': '사용자 ID'})
                st.dataframe(user_stats, hide_index=True, width="stretch")
            
        st.divider()

        # --- 영역 C: 질문 검색 상세 로그 (데이터 클렌징) ---
        st.subheader("📋 전체 상세 로그 테이블")
        
        display_df = df.copy()
        
        col_filt1, col_filt2 = st.columns(2)
        with col_filt1:
            user_filter = st.selectbox("사용자 필터", ["All"] + list(display_df["username"].unique()))
            if user_filter != "All":
                display_df = display_df[display_df["username"] == user_filter]
        with col_filt2:
            search_query = st.text_input("질문 검색 (전체 로그 기준)")
            if search_query:
                display_df = display_df[display_df["query"].str.contains(search_query, case=False, na=False)]

        # 데이터 전처리 및 이름 한글화
        if "generation_model" in display_df.columns:
            display_df["generation_model"] = display_df["generation_model"].fillna("-")
        if "evaluation_model" in display_df.columns:
            display_df["evaluation_model"] = display_df["evaluation_model"].fillna("-")
        if "eval_reason" in display_df.columns:
            display_df["eval_reason"] = display_df["eval_reason"].fillna("-")
        if "context" in display_df.columns:
             display_df["context"] = display_df["context"].fillna("-")
        if "latency" in display_df.columns:
             display_df["latency"] = display_df["latency"].round(2)

        rename_dict = {
            "timestamp": "일시", 
            "username": "사용자",
            "generation_model": "답변 모델",
            "evaluation_model": "심판 모델",
            "query": "사용자 질문", 
            "context": "검색된 문서(요약)", 
            "response": "AI 답변", 
            "latency": "응답속도(초)",
            "generation_score": "평가 점수",
            "eval_reason": "평가 사유(Reason)"
        }
        
        ordered_cols = ["timestamp", "username", "generation_model", "evaluation_model", "query", "context", "response", "latency", "generation_score", "eval_reason"]
        final_cols = [c for c in ordered_cols if c in display_df.columns]
        
        display_df = display_df[final_cols].rename(columns=rename_dict)
        
        def highlight_issues(row):
            latency = row.get("응답속도(초)", 0)
            score = row.get("평가 점수", "평가 대기중")
            
            is_slow = False
            try:
                if float(latency) >= 10.0:
                    is_slow = True
            except Exception: pass
            
            is_fail = False
            try:
                if float(score) == 0.0:
                    is_fail = True
            except Exception: pass
            
            color = 'background-color: rgba(255, 75, 75, 0.2)' if (is_slow or is_fail) else ''
            return [color] * len(row)

        styled_df = display_df.style.apply(highlight_issues, axis=1)
        
        # 소수점 포맷팅 추가 (NaN은 무시하고, 실수형만 지정된 포맷으로 변경)
        styled_df = styled_df.format({
            "응답속도(초)": "{:.2f}",
            "평가 점수": "{:.0f}"
        }, na_rep="-")

        st.dataframe(styled_df, width="stretch", hide_index=True)
                
    else:
        st.info("로그 없음")
