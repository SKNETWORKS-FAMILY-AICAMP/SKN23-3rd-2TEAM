"""
WELD·BOT v3.1 — Professional Arc Flash Edition
Optimized for: FastAPI Backend + LangGraph + GPU Acceleration
"""

import streamlit as st
import requests
import json
import time
import extra_streamlit_components as stx

# ─────────────────────────────────────────────
# 1. Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WELD·BOT — Welding Tech Support",
    page_icon="🔥",
    layout="wide"
)

BACKEND_URL = "http://localhost:8000"

# ─────────────────────────────────────────────
# 2. 통합 CSS (용접 아크 테마 + 소요 시간 애니메이션)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;700;900&display=swap');

:root {
    --bg: #0E1117;
    --surface: #1F2937;
    --accent: #FF6B00; /* 용접 아크 오렌지 */
    --plasma: #00C8FF; /* 플라즈마 블루 */
    --spark: #FFE135;  /* 스파크 옐로 */
    --text: #E5E7EB;
    --mono: 'Share Tech Mono', monospace;
}

/* 전체 배경 및 폰트 */
.stApp { background-color: var(--bg) !important; font-family: 'Barlow Condensed', sans-serif !important; }

/* 채팅 말풍선 - 용접 강철 플레이트 스타일 */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    border-left: 4px solid var(--accent) !important;
}

/* 노드 상태 표시줄 디자인 */
.node-row {
    display: flex; align-items: center; gap: 12px; padding: 8px 0;
    font-family: var(--mono); font-size: 0.85rem; border-bottom: 1px solid #374151;
}

/* 실시간 가동 중 애니메이션 (아크 펄스) */
@keyframes arc-pulse {
    0% { opacity: 1; text-shadow: 0 0 5px var(--plasma); }
    50% { opacity: 0.4; text-shadow: 0 0 15px var(--plasma); }
    100% { opacity: 1; text-shadow: 0 0 5px var(--plasma); }
}
.arc-active { animation: arc-pulse 0.8s infinite; color: var(--plasma) !important; }

/* 타이밍 뱃지 */
.timing-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #111827; border: 1.5px solid var(--spark);
    border-radius: 4px; padding: 6px 12px;
    font-family: var(--mono); color: var(--spark);
    box-shadow: 0 0 10px rgba(255, 225, 53, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. 세션 및 쿠키 관리 (에러 방지용 session_state)
# ─────────────────────────────────────────────
if "cookie_manager" not in st.session_state:
    st.session_state.cookie_manager = stx.CookieManager()
cookie_manager = st.session_state.cookie_manager

# [V3.1] thread_id 초기화 로직
if "thread_id" not in st.session_state:
    saved_tid = cookie_manager.get(cookie="thread_id")
    st.session_state.thread_id = saved_tid if saved_tid else f"weld_worker_{int(time.time())}"
    if not saved_tid:
        cookie_manager.set("thread_id", st.session_state.thread_id, key="init_tid")

# ─────────────────────────────────────────────
# 4. 채팅 루프 및 SSE 시간 측정 연동
# ─────────────────────────────────────────────
st.title("🔥 WELD·BOT v3.1")
st.caption("용접 로봇 기술지원 전용 AI 엔진 가동 중 (FastAPI + LangGraph)")

NODE_DEFS = [
    ("rewriter",   "🔧 쿼리 정규화"),
    ("retriever",  "🔍 매뉴얼 검색"),
    ("reranker",   "⚖️  정밀 재배열"),
    ("specialist", "🤖 전문가 답변"),
    ("verifier",   "🔩 안전 검증")
]

if prompt := st.chat_input("용접 에러 조치법을 물어보세요..."):
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        ans_area = st.empty()
        status_box = st.status("⚡ 아크 발생 중...", expanded=True)
        timing_area = st.empty()
        
        full_ans = ""
        start_time = time.perf_counter()
        timings = {}

        try:
            with requests.post(f"{BACKEND_URL}/chat", 
                               json={"message": prompt, "thread_id": st.session_state.thread_id}, 
                               stream=True, timeout=120) as r:
                active_node = None
                for line in r.iter_lines():
                    if not line: continue
                    data = json.loads(line.decode("utf-8")[6:])
                    
                    dtype = data.get("type")
                    node = data.get("node")
                    
                    # 1. 노드 시작 감지
                    if dtype == "status" and node:
                        active_node = node
                        if node not in timings: timings[node] = {"start": time.perf_counter()}
                    
                    # 2. 노드 종료 및 시간 수신 (백엔드 전송값)
                    elif dtype == "metadata" and node:
                        if node in timings:
                            timings[node]["end"] = time.perf_counter()
                            timings[node]["elapsed"] = data.get("elapsed", 0.0)
                        if active_node == node: active_node = None
                    
                    # 3. 답변 스트리밍
                    elif dtype == "answer":
                        full_ans = data.get("content", "")
                        ans_area.markdown(full_ans + "▌")

                    # 실시간 상태창 업데이트
                    with status_box:
                        html_status = ""
                        for k, label in NODE_DEFS:
                            t = timings.get(k)
                            if t and "end" in t:
                                html_status += f"<div class='node-row'>✅ {label} <span style='margin-left:auto'>{t['elapsed']:.2f}s</span></div>"
                            elif k == active_node:
                                html_status += f"<div class='node-row arc-active'>⚡ {label} <span style='margin-left:auto'>연산 중...</span></div>"
                            else:
                                html_status += f"<div class='node-row' style='color:#4B5563'>○ {label}</div>"
                        st.markdown(html_status, unsafe_allow_html=True)

            ans_area.markdown(full_ans)
            total_time = time.perf_counter() - start_time
            timing_area.markdown(f"<div class='timing-badge'>⚡ 분석 완료 | 총 소요 시간: {total_time:.2f}초</div>", unsafe_allow_html=True)
            status_box.update(label="✅ 분석 완료", state="complete", expanded=False)

        except Exception as e:
            st.error(f"통신 오류: {e}")