# frontend/chat_ui.py
import streamlit as st
import asyncio
import requests
import json
import time
from sseclient import SSEClient

# LangGraph, PostgreSQL 의존성을 모두 제거했습니다! (속도/메모리 완벽 최적화)
# Streamlit은 무거운 로직을 돌리지 않고, FastAPI 백엔드로 API 요청만 넘깁니다.

def generate_agent_response(user_input: str, thread_id: str, user_id: str = "Unknown"):
    """
    FastAPI 백엔드의 /chat/stream SSE 엔드포인트를 호출하여 답변을 스트리밍합니다.
    """
    API_URL = "http://localhost:8000"
    
    # 🌟 FastAPI 백엔드로 POST 요청 전송 (BM25, LangGraph 처리 및 DB 히스토리 저장까지 백엔드에서 모두 수행)
    response = requests.post(
        f"{API_URL}/chat",
        json={"message": user_input, "thread_id": thread_id, "user_id": user_id},
        stream=True
    )
    
    client = SSEClient(response)
    for event in client.events():
        if event.data == "[DONE]":
            break
        
        try:
            data = json.loads(event.data)
            # FastAPI의 sse_event_generator는 다양한 type의 이벤트를 보냅니다.
            # UI 단에서 type에 따라 핸들링하도록 dict 전체를 반환합니다.
            yield data
        except json.JSONDecodeError:
            # JSON 포맷이 아닌 일반 텍스트 스트리밍일 경우 (Fallback)
            yield {"type": "answer", "content": event.data}


def show_chat_page():
    """메인 챗봇 UI를 렌더링하는 함수"""
    st.title("🤖 WELD·BOT 산업 안전 어시스턴트")
    
    # 상단 정보 표시 (세션에서 유저 ID와 권한 가져오기)
    user = st.session_state.get("user") or {}
    user_id = user.get("id", "00000000-0000-0000-0000-000000000000") # UUID 형식의 fallback
    username = user.get("username", "Unknown")
    role = user.get("role", "user")
    st.caption(f"🟢 접속자: **{username}** | 권한: **{role}**")
    if role == "admin" and st.button("Admin UI로 이동", type="secondary"):
        st.session_state.nav_selection = "Admin Dashboard"
        st.rerun()
    st.divider()
    
    # 1. 대화 기록 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 2. 기존 대화 기록 화면에 출력
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. 사용자 입력 처리
    if user_input := st.chat_input("증상, 에러코드, 또는 현장 은어를 입력하세요..."):
        # 사용자 메시지 화면 출력 및 세션 저장
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # 4. AI 스트리밍 답변 처리 (LangGraph 연동)
        with st.chat_message("assistant"):
            # "분석 중..." 로딩 애니메이션 상태 구성 (초기엔 열려있음)
            status_container = st.empty()
            with status_container.status("🔍 챗봇이 답변을 준비하고 있습니다...", expanded=True) as status:
                st.write("요청을 서버로 전송했습니다...")
            
            placeholder = st.empty()
            
            def run_stream():
                full_text = ""
                current_thread_id = f"streamlit_session_{username}"
                
                # FastAPI 백엔드 연동
                for event_data in generate_agent_response(user_input, current_thread_id, user_id):
                    evt_type = event_data.get("type")
                    content = event_data.get("content", "")
                    
                    if evt_type == "status":
                        status.write(content)
                    elif evt_type == "status_complete":
                        status.update(label="✅ 답변 준비 완료", state="complete", expanded=False)
                    elif evt_type == "answer":
                        # 스트리밍이 시작되면 상태바를 닫음 (만약 status_complete가 누락될 경우를 대비)
                        status.update(state="complete", expanded=False)
                        full_text += content
                        # 부드러운 스트리밍 UI 위해 매 청크마다 렌더링
                        placeholder.markdown(full_text + "▌")
                    elif evt_type in ["error", "warning"]:
                        status.write(f"⚠️ {content}")
                        if evt_type == "error":
                            status.update(label="에러 발생", state="error", expanded=True)
                
                # 스트리밍 완료 후 커서 제거
                placeholder.markdown(full_text)
                return full_text
            
            # 스트림 처리 실행
            final_response = run_stream()
            
            # 대화 기록에 최종 답변 저장
            st.session_state.chat_history.append({"role": "assistant", "content": final_response})
