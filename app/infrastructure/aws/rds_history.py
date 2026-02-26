import os
import psycopg2
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

def get_rds_connection():
    """AWS RDS PostgreSQL 연결을 반환합니다."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("RDS_HOST", "localhost"),
            database=os.getenv("RDS_DB", "chatbot_db"),
            user=os.getenv("RDS_USER", "postgres"),
            password=os.getenv("RDS_PASSWORD", "password"),
            port=os.getenv("RDS_PORT", "5432")
        )
        return conn
    except Exception as e:
        print(f"[AWS RDS] 연결 실패: {e}")
        return None

def fetch_conversation_history(thread_id: str) -> List[BaseMessage]:
    """
    AWS RDS에서 특정 thread_id의 전체 대화 내역을 조회하여 
    LangChain 메시지 객체 리스트로 반환합니다.
    """
    print(f"[AWS RDS] 세션 '{thread_id}'의 대화 이력을 불러오는 중...")
    
    conn = get_rds_connection()
    if not conn:
        return []

    messages = []
    try:
        with conn.cursor() as cur:
            # 대화 이력 테이블(예: chat_history)에서 시간순으로 정렬하여 조회
            query = """
                SELECT role, content 
                FROM chat_history 
                WHERE thread_id = %s 
                ORDER BY created_at ASC
            """
            cur.execute(query, (thread_id,))
            rows = cur.fetchall()
            
            for role, content in rows:
                if role.lower() == "human":
                    messages.append(HumanMessage(content=content))
                elif role.lower() == "ai":
                    messages.append(AIMessage(content=content))
        
        print(f"[AWS RDS] 총 {len(messages)}개의 메시지를 불러왔습니다.")
    except Exception as e:
        print(f"[AWS RDS] 데이터 조회 오류: {e}")
    finally:
        conn.close()
        
    return messages
