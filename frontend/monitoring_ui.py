import pandas as pd
import streamlit as st

from app.core.database import get_chat_logs


def show_monitoring_page():
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

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        .block-container {
            max-width: 1600px !important;
            padding: 0 48px 2rem 48px !important;
            margin: 0 auto !important;
        }

        /* ── NAV ── */
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

        .logo { display: flex; align-items: center; gap: 10px; text-decoration: none !important; }
        .logo-dot {
            width: 14px; height: 14px; border-radius: 50%;
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            box-shadow: 0 0 20px rgba(46,197,255,0.8);
        }
        .logo-name {
            font-family: 'Chakra Petch', sans-serif;
            font-size: 0.95rem; font-weight: 650;
            color: #dbf5ff; letter-spacing: 0.08em;
        }

        .nav-online {
            margin-left: 20px;
            font-size: 0.82rem;
            color: #a7f3d0;
            font-weight: 600;
            text-shadow: 0 0 10px rgba(34,197,94,0.28);
            white-space: nowrap;
        }

        .content-spacer { height: calc(var(--nav-h) + 36px); }

        /* ── NAV BUTTONS ── */
        .st-key-mon_nav_home   { position: fixed; top: 16px; right: 306px; width: 80px; z-index: 10020; }
        .st-key-mon_nav_chat   { position: fixed; top: 16px; right: 216px; width: 80px; z-index: 10020; }
        .st-key-mon_nav_admin  { position: fixed; top: 16px; right: 126px; width: 80px; z-index: 10020; }
        .st-key-mon_nav_logout { position: fixed; top: 16px; right: 24px;  width: 94px; z-index: 10020; }

        .st-key-mon_nav_home button,
        .st-key-mon_nav_chat button,
        .st-key-mon_nav_admin button,
        .st-key-mon_nav_logout button {
            width: 100% !important;
            min-height: 32px !important;
            border-radius: 8px !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            border: 1px solid transparent !important;
            background: transparent !important;
        }

        .st-key-mon_nav_home button  { border-color: #fff !important; color: #fff !important; }
        .st-key-mon_nav_home button:hover { background: rgba(255,255,255,0.1) !important; }
        .st-key-mon_nav_chat button  { border-color: var(--secondary) !important; color: var(--secondary) !important; }
        .st-key-mon_nav_chat button:hover { background: rgba(46,197,255,0.1) !important; }
        .st-key-mon_nav_admin button  { border-color: #22c55e !important; color: #22c55e !important; }
        .st-key-mon_nav_admin button:hover { background: rgba(34,197,94,0.1) !important; }
        .st-key-mon_nav_logout button  { border-color: var(--primary) !important; color: var(--primary) !important; }
        .st-key-mon_nav_logout button:hover { background: rgba(255,106,61,0.1) !important; }

        /* ── PAGE TITLE ── */
        .page-title {
            font-family: 'Noto Serif KR', serif;
            font-size: 2.0rem;
            font-weight: 700;
            color: #111;
            text-align: center;
            margin: 0 0 12px 0;
        }
        .page-sub {
            font-family: 'Noto Serif KR', serif;
            font-size: 0.95rem;
            color: #666;
            text-align: center;
            line-height: 1.7;
            margin: 0 0 40px 0;
        }

        /* ── SECTION LABEL ── */
        .section-label {
            font-family: 'Noto Serif KR', serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #111;
            margin: 0 0 16px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid #111;
        }

        /* ── METRIC CARDS ── */
        .metric-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px 22px;
        }
        .metric-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: #999;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .metric-value {
            font-family: 'Chakra Petch', sans-serif;
            font-size: 2.3rem;
            font-weight: 700;
            color: #111;
            line-height: 1;
            margin-bottom: 7px;
        }
        .metric-value .unit { font-size: 1.0rem; font-weight: 400; color: #bbb; }
        .metric-sub { font-size: 0.72rem; color: #aaa; }
        .metric-sub .red   { color: #e53e3e; font-weight: 700; }
        .metric-sub .amber { color: #d97706; font-weight: 700; }

        /* ── WHITE BLOCK ── */
        .block-title {
            font-size: 0.78rem;
            font-weight: 700;
            color: #888;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 16px;
        }

        /* ── LEGEND ── */
        .legend-row { display: flex; gap: 22px; margin-top: 10px; }
        .legend-item { display: flex; align-items: center; gap: 7px; font-size: 0.72rem; color: #888; }
        .legend-dot  { width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }

        /* ── STREAMLIT OVERRIDES ── */
        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.07em !important;
            text-transform: uppercase !important;
            color: #999 !important;
        }
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stTextInput"] > div > div > input {
            background: #fafafa !important;
            border: 1px solid #d8d8d8 !important;
            border-radius: 8px !important;
            color: #111 !important;
            font-size: 0.85rem !important;
        }
        .stDataFrame { border-radius: 10px !important; overflow: hidden !important; }
        .vega-embed { background: transparent !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── NAV BAR ──
    st.markdown(
        f"""
        <div class="nav">
            <a class="logo" href="/?route=home" target="_self">
                <div class="logo-dot"></div>
                <div class="logo-name">WELDPILOT AI</div>
            </a>
            <span class="nav-online">● {username}님 접속 중</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # ── PAGE HEADER ──
    st.markdown(
        """
        <h1 class="page-title">RAG 품질 모니터링</h1>
        <p class="page-sub">AI 심판(LLM-as-a-Judge)이 측정한 시스템 건강 상태 및 오답 감점 사유 로그입니다.</p>
        """,
        unsafe_allow_html=True,
    )

    # ── FETCH DATA ──
    try:
        logs = get_chat_logs(limit=1000)
    except Exception as e:
        st.error(f"DB 연결 오류: {e}")
        logs = []

    if not logs:
        st.markdown(
            "<p style='text-align:center;color:#aaa;padding:60px 0;font-family:Noto Serif KR,serif;'>로그 데이터가 없습니다.</p>",
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(logs)
    if "latency" in df.columns:
        df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
    if "generation_score" in df.columns:
        df["generation_score"] = pd.to_numeric(df["generation_score"], errors="coerce")

    # ── SECTION A: METRICS ──
    st.markdown('<p class="section-label">시스템 헬스 체크 지표</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if "generation_score" in df.columns:
            v = df.dropna(subset=["generation_score"])
            pr = v["generation_score"].mean() * 100 if not v.empty else 0
            fail_n = int((v["generation_score"] == 0).sum())
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">답변 충실도 (Pass Rate)</div>
                    <div class="metric-value">{pr:.1f}<span class="unit">%</span></div>
                    <div class="metric-sub"><span class="{'red' if fail_n > 0 else ''}">실패 {fail_n}건</span> / 전체 {len(v)}건</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c2:
        if "retrieval_total_chunks" in df.columns and "retrieval_relevant_chunks" in df.columns:
            vr = df.dropna(subset=["retrieval_total_chunks"])
            f1 = (vr["retrieval_relevant_chunks"] / vr["retrieval_total_chunks"].replace(0, 1)).mean() * 100 if not vr.empty else 0
            tag = "red" if f1 < 30 else ("amber" if f1 < 60 else "")
            label = "낮음" if f1 < 30 else ("보통" if f1 < 60 else "양호")
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">문서 검색 품질 (F1 Score)</div>
                    <div class="metric-value">{f1:.1f}<span class="unit">%</span></div>
                    <div class="metric-sub"><span class="{tag}">{label}</span></div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="metric-card">
                    <div class="metric-label">문서 검색 품질 (F1 Score)</div>
                    <div class="metric-value" style="font-size:1.4rem;color:#ccc">N/A</div>
                    <div class="metric-sub">청크 데이터 없음</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c3:
        if "latency" in df.columns:
            vl = df.dropna(subset=["latency"])
            avg = vl["latency"].mean() if not vl.empty else 0
            slow_n = int((vl["latency"] >= 15.0).sum())
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">평균 응답 속도</div>
                    <div class="metric-value">{avg:.1f}<span class="unit">s</span></div>
                    <div class="metric-sub"><span class="{'amber' if slow_n > 0 else ''}">15s 초과 {slow_n}건</span></div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c4:
        total_q = len(df)
        uniq_u = df["username"].nunique() if "username" in df.columns else 0
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">총 질문 수</div>
                <div class="metric-value">{total_q}<span class="unit">건</span></div>
                <div class="metric-sub">사용자 {uniq_u}명</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── SECTION B: USAGE ──
    st.markdown('<p class="section-label">시스템 이용 현황</p>', unsafe_allow_html=True)

    col_chart, col_top = st.columns([3, 2], gap="medium")

    with col_chart:
        st.markdown('<div class="white-block"><div class="block-title">일별 질문 추이</div>', unsafe_allow_html=True)
        if "timestamp" in df.columns and not df.empty:
            try:
                import altair as alt
                df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                daily = df.groupby("date").size().reset_index(name="count")
                if not daily.empty:
                    chart = (
                        alt.Chart(daily)
                        .mark_area(
                            line={"color": "#111", "strokeWidth": 2},
                            color=alt.Gradient(
                                gradient="linear",
                                stops=[
                                    alt.GradientStop(color="rgba(0,0,0,0.15)", offset=0),
                                    alt.GradientStop(color="rgba(0,0,0,0.0)", offset=1),
                                ],
                                x1=1, x2=1, y1=1, y2=0,
                            ),
                        )
                        .encode(
                            x=alt.X("date:T", axis=alt.Axis(
                                labelColor="#aaa", gridColor="rgba(0,0,0,0.05)",
                                domainColor="#ddd", tickColor="transparent",
                                labelFontSize=10, format="%m/%d", title=None,
                            )),
                            y=alt.Y("count:Q", axis=alt.Axis(
                                labelColor="#aaa", gridColor="rgba(0,0,0,0.05)",
                                domainColor="transparent", tickColor="transparent",
                                labelFontSize=10, title=None,
                            )),
                            tooltip=[
                                alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                                alt.Tooltip("count:Q", title="질문 수"),
                            ],
                        )
                        .properties(height=220, background="transparent")
                        .configure_view(strokeWidth=0, fill="transparent")
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.markdown("<p style='color:#aaa;font-size:0.8rem;padding:30px 0;text-align:center;'>데이터 없음</p>", unsafe_allow_html=True)
            except Exception:
                df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                daily = df.groupby("date").size().reset_index(name="count")
                st.line_chart(daily.set_index("date"), y="count", height=220)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_top:
        st.markdown('<div class="white-block"><div class="block-title">Top 사용자 현황</div>', unsafe_allow_html=True)
        if "username" in df.columns and "latency" in df.columns:
            us = (
                df.groupby("username")
                .agg(질문건수=("id", "count"), avg_lat=("latency", "mean"))
                .reset_index()
                .sort_values("질문건수", ascending=False)
                .head(10)
            )
            us["평균응답속도(s)"] = us["avg_lat"].round(2)
            us = us.drop(columns=["avg_lat"]).rename(columns={"username": "사용자 ID"})
            st.dataframe(
                us, hide_index=True, use_container_width=True, height=260,
                column_config={
                    "사용자 ID": st.column_config.TextColumn("사용자 ID"),
                    "질문건수": st.column_config.NumberColumn("질문 수"),
                    "평균응답속도(s)": st.column_config.NumberColumn("평균 응답속도(s)", format="%.2f"),
                },
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── SECTION C: LOG TABLE ──
    st.markdown('<p class="section-label">전체 상세 로그 테이블</p>', unsafe_allow_html=True)

    display_df = df.copy()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if "username" in display_df.columns:
            uf = st.selectbox("사용자 필터", ["All"] + sorted(display_df["username"].dropna().unique().tolist()))
            if uf != "All":
                display_df = display_df[display_df["username"] == uf]
    with col_f2:
        sq = st.text_input("질문 검색 (전체 로그 기준)")
        if sq and "query" in display_df.columns:
            display_df = display_df[display_df["query"].str.contains(sq, case=False, na=False)]

    for col in ["generation_model", "evaluation_model", "eval_reason", "context"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].fillna("-")
    if "latency" in display_df.columns:
        display_df["latency"] = display_df["latency"].round(2)

    rename_dict = {
        "timestamp": "일시", "username": "사용자",
        "generation_model": "답변 모델", "evaluation_model": "심판 모델",
        "query": "사용자 질문", "context": "검색된 문서(요약)",
        "response": "AI 답변", "latency": "응답속도(s)",
        "generation_score": "평가 점수", "eval_reason": "평가 사유",
    }
    ordered_cols = ["timestamp", "username", "generation_model", "evaluation_model",
                    "query", "context", "response", "latency", "generation_score", "eval_reason"]
    final_cols = [c for c in ordered_cols if c in display_df.columns]
    display_df = display_df[final_cols].rename(columns=rename_dict)

    # 하이라이트: 빨간=점수 0 / 노란=지연 15s 이상
    def highlight_issues(row):
        score = row.get("평가 점수", None)
        lat   = row.get("응답속도(s)", 0)
        is_fail = False
        is_slow = False
        try:
            if float(score) == 0.0:
                is_fail = True
        except (TypeError, ValueError):
            pass
        try:
            if not is_fail and float(lat) >= 15.0:
                is_slow = True
        except (TypeError, ValueError):
            pass
        if is_fail:
            c = "background-color: rgba(229,62,62,0.10)"
        elif is_slow:
            c = "background-color: rgba(217,119,6,0.08)"
        else:
            c = ""
        return [c] * len(row)

    styled = display_df.style.apply(highlight_issues, axis=1)
    styled = styled.format({"응답속도(s)": "{:.2f}", "평가 점수": "{:.0f}"}, na_rep="-")

    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    st.markdown(
        """
        <div class="legend-row">
            <div class="legend-item">
                <div class="legend-dot" style="background:rgba(229,62,62,0.55);"></div>
                평가 점수 = 0 (응답 실패)
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background:rgba(217,119,6,0.55);"></div>
                응답속도 ≥ 15s (응답 지연)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)