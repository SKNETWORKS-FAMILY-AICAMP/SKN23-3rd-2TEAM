# -*- coding: utf-8 -*-
# WELD-BOT 채팅 UI - 버튼 완전 고정 버전

import html
import json
import os
import time
from datetime import datetime
import extra_streamlit_components as stx
import requests
import streamlit as st
from sseclient import SSEClient
import markdown
import base64

# -------------------------------------------------
# 1. 페이지 설정 (standalone 실행 시에만 사용)
# -------------------------------------------------
def _set_page_config_for_standalone():
    st.set_page_config(
        page_title="WELD-BOT - Welding Tech Support",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

# -------------------------------------------------
# 2. 세션 초기화
# -------------------------------------------------
cookie_manager = None


def _ensure_chat_session_state():
    global cookie_manager

    # Single source of truth: authenticated
    st.session_state.logged_in = bool(st.session_state.get("authenticated", False))
    if "user" not in st.session_state or not isinstance(st.session_state.user, dict):
        st.session_state.user = {"username": "민정", "role": "user"}
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = True
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = {}
    if "thread_last_active" not in st.session_state:
        st.session_state.thread_last_active = {}
    if "pending_chat_request" not in st.session_state:
        st.session_state.pending_chat_request = None
    if "is_generating_response" not in st.session_state:
        st.session_state.is_generating_response = False

    # Single source of truth: reusing manager from v4_app.py session_state
    cookie_manager = st.session_state.get("cookie_manager_auth")
    if not cookie_manager:
        # Fallback if somehow not initialized, but v4_app should handle this
        try:
            cookie_manager = stx.CookieManager(key="weld_cookie_manager_auth")
            st.session_state.cookie_manager_auth = cookie_manager
        except Exception:
            pass
    
    st.session_state.cookie_manager = cookie_manager

    if "thread_id" not in st.session_state:
        saved_tid = cookie_manager.get(cookie="thread_id") if cookie_manager else None
        if saved_tid:
            st.session_state.thread_id = saved_tid
        else:
            new_tid = f"weld_{int(time.time())}"
            st.session_state.thread_id = new_tid
            if cookie_manager:
                try:
                    cookie_manager.set("thread_id", new_tid)
                except Exception:
                    pass

    if st.session_state.thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads[st.session_state.thread_id] = list(st.session_state.messages)
    else:
        st.session_state.messages = list(st.session_state.chat_threads[st.session_state.thread_id])

    if st.session_state.thread_id not in st.session_state.thread_last_active:
        st.session_state.thread_last_active[st.session_state.thread_id] = time.time()


def _sync_current_thread_messages():
    st.session_state.chat_threads[st.session_state.thread_id] = list(st.session_state.messages)

def _mark_current_thread_active():
    st.session_state.thread_last_active[st.session_state.thread_id] = time.time()

def _switch_thread(thread_id: str):
    _sync_current_thread_messages()
    st.session_state.thread_id = thread_id
    if cookie_manager:
        try:
            cookie_manager.set("thread_id", thread_id)
        except Exception:
            pass
    st.session_state.messages = list(st.session_state.chat_threads.get(thread_id, []))


def _get_api_url() -> str:
    env_api_url = os.getenv("API_URL")
    if env_api_url:
        return env_api_url
    try:
        return st.secrets["API_URL"]
    except Exception:
        return "http://localhost:8000"


def generate_agent_response(user_input: str, thread_id: str, user_id: str):
    api_url = _get_api_url()
    response = None

    try:
        response = requests.post(
            f"{api_url}/chat",
            json={"message": user_input, "thread_id": thread_id, "user_id": user_id},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        yield {"type": "error", "content": "백엔드 서버에 연결할 수 없습니다. 서버 실행 상태를 확인해주세요."}
        return
    except requests.exceptions.HTTPError as e:
        yield {"type": "error", "content": f"서버 오류가 발생했습니다: {e.response.status_code}"}
        return
    except requests.exceptions.RequestException as e:
        yield {"type": "error", "content": f"요청 중 문제가 발생했습니다: {str(e)}"}
        return

    try:
        client = SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break
            try:
                yield json.loads(event.data)
            except json.JSONDecodeError:
                yield {"type": "answer", "content": event.data}
    except requests.exceptions.ReadTimeout:
        yield {
            "type": "error",
            "content": "응답 생성 시간이 길어 연결이 시간 초과되었습니다. 잠시 후 다시 시도해주세요.",
        }
    except requests.exceptions.RequestException as e:
        yield {"type": "error", "content": f"스트리밍 연결 오류: {str(e)}"}
    except Exception as e:
        yield {"type": "error", "content": f"스트리밍 처리 중 오류: {str(e)}"}
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


# -------------------------------------------------
# 3. 스타일
# -------------------------------------------------
def inject_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;700&family=Chakra+Petch:wght@400;500;600;700&family=IBM+Plex+Sans+KR:wght@400;500;700&family=Noto+Serif+KR:wght@400;500;700&display=swap');

:root {
  --primary: #ff6a3d;
  --secondary: #2ec5ff;
  --line: rgba(120,190,220,0.20);
  --nav-h: 64px;
  --page-pad: 24px;
  --input-h: 129px;
  --btn-h: 58px;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
.stApp { background: #f2f2f2; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* 본문: 상단바 + 하단 버튼+입력창 높이만큼 여백 */
[data-testid="stMainBlockContainer"] {
  padding-left: var(--page-pad) !important;
  padding-right: var(--page-pad) !important;
  padding-top: calc(var(--nav-h) + 20px) !important;
  padding-bottom: calc(var(--input-h) + var(--btn-h) + 24px) !important;
  max-width: 900px !important;
  margin: 0 auto !important;
}

/* ── 상단바 ── */
.nav {
  display: flex; align-items: center; gap: 20px;
  height: var(--nav-h); position: fixed; top: 0; left: 0; right: 0;
  padding: 0 32px 0 24px; background: #000;
  border-bottom: 1px solid var(--line); z-index: 9998;
}
.logo { display: flex; align-items: center; gap: 10px; font-family: 'Chakra Petch', sans-serif; }
.logo-dot {
  width: 14px; height: 14px; border-radius: 50%;
  background: linear-gradient(135deg, var(--secondary), var(--primary));
  box-shadow: 0 0 20px rgba(46,197,255,0.8);
}
.logo-name { font-size: 0.95rem; font-weight: 600; color: #dbf5ff; letter-spacing: 0.08em; }
.menu { display: flex; gap: 20px; margin-left: auto; align-items: center; }
.btn-logout {
  background: transparent; color: #ff6a3d !important; padding: 6px 14px;
  border-radius: 6px; border: 1px solid #ff6a3d; font-weight: 700;
  box-shadow: 0 0 8px rgba(255,106,61,0.3); transition: all 0.2s;
  text-decoration: none; font-size: 0.84rem; font-family: 'IBM Plex Sans KR', sans-serif;
}
.btn-logout:hover {
  background: rgba(255,106,61,0.1); transform: translateY(-1px);
  box-shadow: 0 0 12px rgba(255,106,61,0.6);
}
.btn-admin {
  background: transparent; color: #22c55e !important; padding: 6px 14px;
  border-radius: 6px; border: 1px solid #22c55e; font-weight: 700;
  box-shadow: 0 0 8px rgba(34,197,94,0.3); transition: all 0.2s;
  text-decoration: none; font-size: 0.84rem; font-family: 'IBM Plex Sans KR', sans-serif;
}
.btn-admin:hover {
  background: rgba(34,197,94,0.1); transform: translateY(-1px);
  box-shadow: 0 0 12px rgba(34,197,94,0.6);
}
.is-hidden { display: none !important; }

.st-key-chat_nav_home {
  position: fixed;
  top: 15px;
  right: 220px;
  width: 80px;
  z-index: 10020;
}
.st-key-chat_nav_admin {
  position: fixed;
  top: 15px;
  right: 132px;
  width: 80px;
  z-index: 10020;
}
.st-key-chat_nav_logout {
  position: fixed;
  top: 15px;
  right: 24px;
  width: 100px;
  z-index: 10020;
}
.st-key-chat_nav_home button,
.st-key-chat_nav_admin button,
.st-key-chat_nav_logout button {
  width: 100% !important;
  min-height: 34px !important;
  border-radius: 7px !important;
  font-size: 0.78rem !important;
  font-weight: 700 !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  color: #fff !important;
}
.st-key-chat_nav_home button {
  border-color: #2ec5ff !important;
  color: #2ec5ff !important;
  box-shadow: 0 0 8px rgba(46,197,255,0.35) !important;
}
.st-key-chat_nav_home button:hover {
  background: rgba(46,197,255,0.12) !important;
  box-shadow: 0 0 12px rgba(46,197,255,0.55) !important;
  color: #8be3ff !important;
}
.st-key-chat_nav_admin button {
  border-color: #22c55e !important;
  color: #22c55e !important;
  box-shadow: 0 0 8px rgba(34,197,94,0.32) !important;
}
.st-key-chat_nav_admin button:hover {
  background: rgba(34,197,94,0.12) !important;
  box-shadow: 0 0 12px rgba(34,197,94,0.55) !important;
  color: #7ee2a4 !important;
}
.st-key-chat_nav_logout button {
  border-color: #ff6a3d !important;
  color: #ff6a3d !important;
  box-shadow: 0 0 8px rgba(255,106,61,0.34) !important;
}
.st-key-chat_nav_logout button:hover {
  background: rgba(255,106,61,0.12) !important;
  box-shadow: 0 0 12px rgba(255,106,61,0.6) !important;
  color: #ff9b7c !important;
}

.chat-online-status {
  position: fixed;
  top: 21px;
  right: 340px;
  z-index: 10025;
  color: #a7f3d0;
  font-size: 0.83rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  pointer-events: none;
  text-shadow: 0 0 10px rgba(34,197,94,0.28);
}

/* ══════════════════════════════════════════════════
   하단 고정 레이아웃
══════════════════════════════════════════════════ */
[data-testid="stBottomBlockContainer"] {
  position: fixed !important;
  bottom: 0 !important;
  left: 0 !important;
  right: 0 !important;
  width: 100vw !important;
  padding: 28px 0 28px !important;
  background: #ffffff !important;
  box-sizing: border-box !important;
  z-index: 998 !important;
  display: flex !important;
  align-items: center !important;
}
[data-testid="stBottomBlockContainer"] > div {
  max-width: 900px !important;
  margin: 0 auto !important;
  padding: 0 24px !important;
}
[data-testid="stBottom"] {
  all: unset !important;
  display: block !important;
}
[data-testid="stBottom"] > div { padding: 0 !important; }

[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
  position: fixed !important;
  bottom: 96px !important;
  left: 0 !important;
  right: 0 !important;
  width: 100vw !important;
  padding: 12px calc(50vw - 426px) !important;
  background: #ffffff !important;
  border-top: 1px solid #e5e7eb !important;
  box-sizing: border-box !important;
  z-index: 999 !important;
  display: flex !important;
  gap: 12px !important;
  margin: 0 !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
  flex: 1 !important;
  min-width: 0 !important;
  padding: 0 !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] button {
  width: 100% !important;
  height: 44px !important;
  border-radius: 10px !important;
  border: none !important;
  background: #111827 !important;
  color: #ffffff !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  font-family: 'IBM Plex Sans KR', sans-serif !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
  transition: all 0.18s ease !important;
  cursor: pointer !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] button:hover {
  background: #1f2937 !important;
  box-shadow: 0 4px 14px rgba(0,0,0,0.25) !important;
  transform: translateY(-1px) !important;
}

/* ── 채팅 버블 ── */
.page-hero { padding: 20px 0 12px; text-align: center; }
.headline {
  font-family: 'Archivo', sans-serif;
  font-size: clamp(1.8rem,3vw,2.2rem);
  color: #000; margin-bottom: 6px;
}
.sub { font-family: 'Noto Serif KR', serif; color: #4b5563; font-size: 0.95rem; }

.chat-wrap { padding: 0; }
.chat-row { display: flex; flex-direction: column; margin: 1rem 0; max-width: 75%; }
.assistant-row { margin-right: auto; align-items: flex-start; }
.user-row { margin-left: auto; align-items: flex-end; }
.chat-meta { font-size: 0.85rem; color: #6b7280; font-weight: 600; margin-bottom: 4px; }

.date-divider-wrap { display: flex; justify-content: center; margin: 30px 0 20px; }
.date-divider {
  background: #e5e7eb; color: #6b7280;
  font-size: 0.8rem; font-weight: 600;
  padding: 6px 16px; border-radius: 20px;
}
.msg-time { font-size: 0.75rem; color: #9ca3af; margin-top: 6px; }
.user-row .msg-time { text-align: right; margin-right: 6px; }
.assistant-row .msg-time { text-align: left; margin-left: 6px; }

.chat-bubble {
  background: #fff; 
  border-radius: 20px; 
  padding: 10px 16px; 
  border: 1px solid rgba(0,0,0,0.08); 
  box-shadow: 0 4px 16px rgba(0,0,0,0.05);
  color: #0b1116; 
  position: relative;
  width: fit-content; 
  word-break: break-word;
}
.user-row .chat-bubble { border-top-right-radius: 4px; }
.assistant-row .chat-bubble { border-top-left-radius: 4px; }

/* ── 마크다운 텍스트 및 헤더(제목) 크기 조정 ── */
.chat-text {
  font-size: 0.9rem; /* 전체 본문 글자 크기를 살짝 줄임 */
  line-height: 1.6;
}
.chat-text p {
  margin: 0 0 8px 0 !important; /* 문단 간격 조정 */
}
.chat-text p:last-child {
  margin-bottom: 0 !important; /* 마지막 문단은 여백 제거 */
}
/* 제목(##, ### 등)이 너무 커지지 않도록 강제 고정 */
.chat-text h1 { font-size: 1.15rem; margin: 12px 0 6px; font-weight: 700; color: #111; }
.chat-text h2 { font-size: 1.05rem; margin: 12px 0 6px; font-weight: 700; color: #111; }
.chat-text h3 { font-size: 0.95rem; margin: 10px 0 6px; font-weight: 700; color: #111; }
.chat-text h4, .chat-text h5, .chat-text h6 { font-size: 0.9rem; margin: 8px 0 4px; font-weight: 700; }

/* ── 리스트(목록) 스타일 조정 ── */
.chat-text ul, .chat-text ol {
  margin: 4px 0 10px 20px;
  padding: 0;
}
.chat-text li {
  margin-bottom: 4px;
}

/* ── 마크다운 표(Table) 컴팩트 스타일 ── */
.chat-bubble table {
  border-collapse: collapse;
  width: 100%;
  margin-top: 8px;
  margin-bottom: 12px;
  font-size: 0.85rem; /* 표 안의 글자 크기를 더 작게 */
}
.chat-bubble th, .chat-bubble td {
  border: 1px solid #d1d5db;
  padding: 6px 10px; /* 표 셀 안쪽 여백 축소 */
  text-align: left;
  line-height: 1.4;
}
.chat-bubble th {
  background-color: #f3f4f6;
  font-weight: 700;
  color: #333;
}

.chat-profile-img {
  width: 26px;
  height: 26px;
  border-radius: 50%; /* 동그랗게 */
  margin-right: 8px;
  vertical-align: middle; /* 텍스트랑 높이 맞춤 */
  object-fit: cover;
  border: 1px solid rgba(0,0,0,0.1); /* 살짝 테두리 */
}

.chat-meta {
  display: flex;
  align-items: center; /* 아이콘과 글자 세로 중앙 정렬 */
  font-size: 0.85rem;
  color: #6b7280;
  font-weight: 600;
  margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# 4. 상단바 및 모달
# -------------------------------------------------
def render_navbar():
    def _do_logout() -> None:
        st.session_state.logged_in = False
        st.session_state.messages = []
        st.session_state.user = None
        st.session_state.authenticated = False
        st.session_state.access_token = None
        st.session_state.force_logged_out = True
        st.session_state.cookie_restore_attempted = False
        st.session_state.cookie_restore_attempt_count = 0
        st.session_state.auth_route = "home"

        controller = st.session_state.get("cookie_controller")
        if controller:
            try:
                controller.remove("weld_access_token", path="/", same_site="lax")
                controller.set("weld_access_token", "", max_age=0, path="/")
            except Exception:
                pass
        backup_manager = st.session_state.get("cookie_manager_auth")
        if backup_manager:
            try:
                backup_manager.delete("weld_access_token")
            except Exception:
                pass

        import time
        st.query_params.clear()
        st.query_params["public"] = "main"
        time.sleep(0.5)
        st.rerun()

    params = st.query_params
    if params.get("action") == "logout":
        _do_logout()

    user = st.session_state.get("user")
    if not isinstance(user, dict):
        user = {}
    username = user.get("username", "민정")
    role = user.get("role", "user")
    is_admin = role == "admin"

    # Role-aware top-right layout tuning:
    # - Admin: Home + Admin + Logout
    # - User: Home + Logout (Home shifts right to remove visual gap)
    home_right = "220px" if is_admin else "132px"
    status_right = "340px" if is_admin else "252px"

    # Keep Home color consistent across roles.
    home_color = "#2ec5ff"
    home_glow = "rgba(46,197,255,0.35)"
    home_hover_bg = "rgba(46,197,255,0.12)"
    home_hover_glow = "rgba(46,197,255,0.55)"
    home_hover_text = "#8be3ff"

    st.markdown(
        f"""
        <style>
        .st-key-chat_nav_home {{
          right: {home_right} !important;
        }}
        .chat-online-status {{
          right: {status_right} !important;
        }}
        .st-key-chat_nav_home button {{
          border-color: {home_color} !important;
          color: {home_color} !important;
          box-shadow: 0 0 8px {home_glow} !important;
        }}
        .st-key-chat_nav_home button:hover {{
          background: {home_hover_bg} !important;
          box-shadow: 0 0 12px {home_hover_glow} !important;
          color: {home_hover_text} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="nav">
      <div style="display:flex;align-items:center;">
        <div class="logo">
          <div class="logo-dot"></div>
          <div class="logo-name">WELDPILOT AI</div>
        </div>
      </div>
      <div class="menu"></div>
    </div>
    """, unsafe_allow_html=True)

    username = st.session_state.get("user_name", "Unknown")
    st.markdown(
        f'<div class="chat-online-status">🟢 {username}님 접속 중</div>',
        unsafe_allow_html=True,
    )

    if st.button("Home", key="chat_nav_home", use_container_width=True):
        st.query_params["route"] = "home"
        st.session_state.auth_route = "home"
        st.rerun()
 
    if is_admin:
        if st.button("Admin", key="chat_nav_admin", use_container_width=True):
            st.query_params["route"] = "settings"
            st.session_state.auth_route = "settings"
            st.rerun()

    if st.button("Logout", key="chat_nav_logout", use_container_width=True):
        _do_logout()


@st.dialog("📋 내 채팅 목록")
def show_chat_list_modal():
    st.write("이전에 대화했던 내역을 불러옵니다.")
    st.divider()

    ordered_threads = sorted(
        st.session_state.chat_threads.keys(),
        key=lambda tid: st.session_state.thread_last_active.get(tid, 0),
        reverse=True,
    )

    count = 0
    for tid in ordered_threads:
        thread_msgs = st.session_state.chat_threads.get(tid, [])
        first_user = next((m["content"] for m in thread_msgs if m["role"] == "user"), None)
        if not first_user:
            continue
        preview = first_user[:30] + "..." if len(first_user) > 30 else first_user
        if tid == st.session_state.thread_id:
            st.button(f"현재 💬 {preview}", key=f"modal_current_{tid}", use_container_width=True, disabled=True)
        elif st.button(preview, key=f"modal_switch_{tid}", use_container_width=True):
            _switch_thread(tid)
            st.rerun()
        count += 1
        if count >= 15:
            break

    if count == 0:
        st.info("저장된 대화 내역이 없습니다.")


# -------------------------------------------------
# 5. 메인 채팅 UI
# -------------------------------------------------
def render_chat():
    st.markdown("""
    <div class="page-hero">
      <h1 class="headline">WELD-BOT Live Chat</h1>
      <p class="sub">현장 이슈를 빠르게 정리하고 안전한 로봇용접 가이드를 제공합니다.</p>
    </div>
    """, unsafe_allow_html=True)

    is_busy = bool(st.session_state.get("pending_chat_request")) or bool(
        st.session_state.get("is_generating_response")
    )

    def get_image_base64(image_path):
        """이미지 파일을 Base64 문자열로 변환 (수정됨: 헤더 추가)"""
        try:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{encoded_string}"
        except Exception:
            # 에러 메시지 팝업 대신 조용히 처리
            return None

    # ── 버튼을 chat_input보다 훨씬 앞에 선언 → stBottom과 완전 분리 ──
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "➕ 새 채팅",
            key="btn_new_chat",
            use_container_width=True,
            disabled=is_busy,
        ):
            _sync_current_thread_messages()
            new_tid = f"weld_{int(time.time())}"
            st.session_state.thread_id = new_tid
            st.session_state.chat_threads[new_tid] = []
            st.session_state.thread_last_active[new_tid] = time.time()
            st.session_state.messages = []
            if cookie_manager:
                try:
                    cookie_manager.set("thread_id", new_tid)
                except Exception:
                    pass
            st.rerun()
    with col2:
        if st.button(
            "📋 채팅 목록",
            key="btn_chat_list",
            use_container_width=True,
            disabled=is_busy,
        ):
            show_chat_list_modal()

    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    last_date_str = None

    # 반복문 밖에서 이미지 한 번만 불러오기 (성능 개선)
    bot_img_base64 = get_image_base64("/Users/jy/3rd-2TEAM/SKN23-3rd-2TEAM/frontend/image/image.png") 
    user_img_base64 = get_image_base64("/Users/jy/3rd-2TEAM/SKN23-3rd-2TEAM/frontend/image/image.png")

    for idx, msg in enumerate(st.session_state.messages):
        role_class = "user-row" if msg["role"] == "user" else "assistant-row"
        
        # 불러온 이미지 데이터 사용
        if msg["role"] == "user":
            icon_html = f'<img src="{user_img_base64}" class="chat-profile-img">' if user_img_base64 else "🧑‍🔧"
            label = f"{icon_html} User"
        else:
            icon_html = f'<img src="{bot_img_base64}" class="chat-profile-img">' if bot_img_base64 else "🤖"
            label = f"{icon_html} Chatbot"
            
        msg_time = msg.get("timestamp", time.time())
        dt_obj = datetime.fromtimestamp(msg_time)
        current_date_str = dt_obj.strftime("%Y/%m/%d")
        ampm = "오후" if dt_obj.hour >= 12 else "오전"
        hour12 = dt_obj.hour % 12 or 12
        time_str = f"{ampm} {hour12}:{dt_obj.strftime('%M')}"

        if current_date_str != last_date_str:
            st.markdown(f"""
            <div class="date-divider-wrap">
              <div class="date-divider">{current_date_str}</div>
            </div>""", unsafe_allow_html=True)
            last_date_str = current_date_str

        content_for_display = markdown.markdown(
            msg["content"], 
            extensions=['tables', 'nl2br', 'fenced_code']
        )

        st.markdown(f"""
        <div class="chat-row {role_class}">
          <div class="chat-meta">{label}</div>
          <div class="chat-bubble">
            <div class="chat-text">{content_for_display}</div>
          </div>
          <div class="msg-time">{time_str}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    user = st.session_state.get("user")
    if not isinstance(user, dict):
        user = {}
    user_id = user.get("id", "00000000-0000-0000-0000-000000000000")

    pending_request = st.session_state.get("pending_chat_request")
    if pending_request and not st.session_state.get("is_generating_response", False):
        st.session_state.is_generating_response = True
        request_prompt = pending_request.get("prompt", "")
        request_thread_id = pending_request.get("thread_id", st.session_state.thread_id)
        request_user_id = pending_request.get("user_id", user_id)

        full_text = ""
        error_text = None

        try:
            with st.spinner("답변을 생성 중입니다..."):
                for event_data in generate_agent_response(request_prompt, request_thread_id, request_user_id):
                    evt_type = event_data.get("type")
                    content = event_data.get("content", "")
                    if evt_type == "answer":
                        full_text += content
                    elif evt_type == "error":
                        error_text = content
                        break

            if error_text:
                full_text = f"오류: {error_text}"
            elif not full_text.strip():
                full_text = "응답을 생성하지 못했습니다. 잠시 후 다시 시도해주세요."

            assistant_msg = {
                "role": "assistant",
                "content": full_text,
                "timestamp": time.time(),
            }

            if st.session_state.thread_id == request_thread_id:
                st.session_state.messages.append(assistant_msg)
                _sync_current_thread_messages()
            else:
                st.session_state.chat_threads.setdefault(request_thread_id, []).append(assistant_msg)

            st.session_state.thread_last_active[request_thread_id] = time.time()
        finally:
            st.session_state.pending_chat_request = None
            st.session_state.is_generating_response = False

        st.rerun()

    # ── 입력창 (stBottom 1개만 생성) ──
    if prompt := st.chat_input("로봇용접 관련 조치법을 입력해주세요.", disabled=is_busy):
        current_time = time.time()
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
        _sync_current_thread_messages()
        _mark_current_thread_active()
        st.session_state.pending_chat_request = {
            "prompt": prompt,
            "thread_id": st.session_state.thread_id,
            "user_id": user_id,
        }
        st.rerun()


# -------------------------------------------------
# 6. 페이지 엔트리포인트
# -------------------------------------------------
def show_chat_page():
    _ensure_chat_session_state()

    if not st.session_state.get("authenticated", False):
        st.markdown('<meta http-equiv="refresh" content="0; url=/">', unsafe_allow_html=True)
        st.stop()

    inject_styles()
    render_navbar()
    render_chat()


if __name__ == "__main__":
    _set_page_config_for_standalone()
    show_chat_page()