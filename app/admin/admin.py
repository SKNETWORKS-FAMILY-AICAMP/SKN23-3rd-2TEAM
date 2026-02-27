import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import sys

# 프로젝트 루트 추가
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from app.core.config import ADMIN_SECRET_KEY, DATA_DIR

# 1. 페이지 설정 및 테마 (Industrial Dark)
st.set_page_config(page_title="RAG Admin Dashboard", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    .stButton>button { color: #ffffff; background-color: #ff4b4b; border-radius: 5px; }
    .stTextInput>div>div>input { background-color: #262730; color: white; }
    .sidebar .sidebar-content { background-color: #161b22; }
    h1, h2, h3 { color: #00d4ff; font-family: 'Inter', sans-serif; }
    .css-1v0mbuf { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# 2. 보안 계층 (Authentication)
def check_password():
    """Returns True if the user had the correct password."""
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if st.session_state.admin_authenticated:
        return True

    st.title("🔐 Admin Access Control")
    password = st.text_input("Admin Secret Key", type="password")
    if st.button("Login"):
        if password == ADMIN_SECRET_KEY:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("❌ Invalid Secret Key")
    return False

if not check_password():
    st.stop()

# 3. 메인 대시보드
st.title("🕵️ RAG 관리자 대시보드")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Dashboard Overview", "Jargon DB Editor", "Fallback Analysis"])

# --- Dashboard Overview ---
if menu == "Dashboard Overview":
    st.header("시스템 상태 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("System Mode", "Industrial Dark")
    col2.metric("DB Type", "AWS RDS (pgvector)")
    col3.metric("Search Mode", "Hybrid (Vector + BM25)")
    
    st.info("실시간 서버 로그 및 리소스 사용량 모니터링 기능 준비 중...")

# --- Jargon DB Editor ---
elif menu == "Jargon DB Editor":
    st.header("📝 현장 은어(Jargon) 사전 관리 (AWS RDS)")
    st.write("작업자들이 사용하는 현장 은어를 표준 기술 용어로 매핑합니다. (DB 테이블: `jargon`)")

    import psycopg2
    
    # DB 연결 정보 (환경 변수 로드)
    ssh_enabled = os.getenv("SSH_TUNNEL_ENABLED", "false").lower() == "true"
    ssh_local_port = os.getenv("SSH_LOCAL_BIND_PORT", "15432")
    target_host = "127.0.0.1" if ssh_enabled else os.getenv("PGHOST", "localhost")
    target_port = ssh_local_port if ssh_enabled else os.getenv("PGPORT", "5432")

    def get_db_connection():
        return psycopg2.connect(
            host=target_host,
            database=os.getenv("PGDATABASE", "chatbot_db"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "password"),
            port=target_port
        )

    # 데이터 불러오기
    try:
        conn = get_db_connection()
        df_jargon = pd.read_sql("SELECT * FROM jargon", conn)
        conn.close()
        
        # 데이터 출력 및 편집
        st.info(f"현재 {len(df_jargon)}개의 은어가 등록되어 있습니다.")
        edited_df = st.data_editor(df_jargon, num_rows="dynamic", width="stretch", key="jargon_editor")
        
        if st.button("Save Changes to AWS RDS"):
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    # 단순화를 위해 전체 삭제 후 재삽입 (데이터량이 적을 때 유효)
                    cur.execute("DELETE FROM jargon")
                    for _, row in edited_df.iterrows():
                        cur.execute(
                            "INSERT INTO jargon (category, jargon, standard) VALUES (%s, %s, %s)",
                            (row['category'], row['jargon'], row['standard'])
                        )
                conn.commit()
                conn.close()
                st.success("✅ AWS RDS 은어 사전이 성공적으로 업데이트되었습니다!")
            except Exception as e:
                st.error(f"❌ DB 저장 오류: {e}")
                
    except Exception as e:
        st.error(f"❌ AWS RDS 연결 오류: {e}")
        st.warning("SSH 터널링이 켜져 있는지, 혹은 DB 접속 정보가 올바른지 확인해 주세요.")

# --- Fallback Analysis ---
elif menu == "Fallback Analysis":
    st.header("📉 검색 실패(Fallback) 쿼리 분석")
    st.write("RAG 검색에서 임계값을 통과하지 못해 Fallback으로 넘어간 쿼리들을 분석합니다.")
    
    FALLBACK_LOG = DATA_DIR / "fallback_queries.jsonl"
    
    if not FALLBACK_LOG.exists():
        st.warning("기록된 Fallback 쿼리가 없습니다.")
    else:
        try:
            fallbacks = []
            with open(FALLBACK_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    fallbacks.append(json.loads(line))
            
            df_fallback = pd.DataFrame(fallbacks)
            
            st.subheader("미답변 쿼리 리스트")
            st.dataframe(df_fallback, width="stretch")
            
            # 간단한 시각화 (도메인별 실패 빈도 등)
            if "category" in df_fallback.columns:
                st.subheader("도메인별 Fallback 비율")
                domain_counts = df_fallback["category"].value_counts()
                st.bar_chart(domain_counts)
        except Exception as e:
            st.error(f"로그 로드 중 오류: {e}")

st.sidebar.divider()
if st.sidebar.button("Logout"):
    st.session_state.admin_authenticated = False
    st.rerun()
