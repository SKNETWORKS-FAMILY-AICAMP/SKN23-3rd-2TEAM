# frontend/chat_ui.py
import streamlit as st
import asyncio
import requests
import json
from sseclient import SSEClient

# LangGraph, PostgreSQL 의존성을 모두 제거했습니다! (속도/메모리 완벽 최적화)
# Streamlit은 무거운 로직을 돌리지 않고, FastAPI 백엔드로 API 요청만 넘깁니다.

def generate_agent_response(user_input: str, thread_id: str):
    """
    FastAPI 백엔드의 /chat/stream SSE 엔드포인트를 호출하여 답변을 스트리밍합니다.
    """
    API_URL = "http://localhost:8000"
    
    # 🌟 FastAPI 백엔드로 POST 요청 전송 (BM25, LangGraph 처리 및 DB 히스토리 저장까지 백엔드에서 모두 수행)
    response = requests.post(
        f"{API_URL}/chat",
        json={"message": user_input, "thread_id": thread_id},
        stream=True
    )
    
    client = SSEClient(response)
    for event in client.events():
        if event.data == "[DONE]":
            break
        
        try:
            data = json.loads(event.data)
            # FastAPI의 sse_event_generator는 다양한 type의 이벤트를 보냅니다.
            # 사용자에게 보여줄 실제 답변 청크만 필터링합니다.
            if data.get("type") == "answer":
                yield data.get("content", "")
            elif data.get("type") in ["error", "warning"]:
                yield f"\n\n**[{data.get('type').upper()}]** {data.get('content', '')}\n"
        except json.JSONDecodeError:
            # JSON 포맷이 아닌 일반 텍스트 스트리밍일 경우 (Fallback)
            yield event.data


def show_chat_page():
    """메인 챗봇 UI를 렌더링하는 함수"""
    st.title("🤖 WELD·BOT 산업 안전 어시스턴트")
    
    # 상단 정보 표시 (세션에서 유저 ID와 권한 가져오기)
    user_id = st.session_state.get('user_id', 'Unknown')
    role = st.session_state.get('role', 'user')
    st.caption(f"🟢 접속자: **{user_id}** | 권한: **{role}**")
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
            # API 스트리밍 호출 및 렌더링
            def run_stream():
                full_text = ""
                placeholder = st.empty()
                current_thread_id = f"streamlit_session_{user_id}"
                
                # FastAPI 백엔드 연동 (매우 빠름, SSH/DB 재연결 불필요)
                for chunk in generate_agent_response(user_input, current_thread_id):
                    full_text += chunk
                    placeholder.markdown(full_text + "▌") 
                
                placeholder.markdown(full_text)
                return full_text
            
            # 동기 함수 실행 (requests.get은 동기이므로 asyncio.run 불필요)
            final_response = run_stream()
            
            # 대화 기록에 최종 답변 저장
            st.session_state.chat_history.append({"role": "assistant", "content": final_response})