from langgraph.checkpoint.memory import MemorySaver

def get_memory_saver():
    """
    세션별(thread_id) 대화 기록을 저장하는 Checkpointer입니다.
    현재는 로컬 인메모리(MemorySaver)를 사용하지만, 추후 필요시
    AWS RDS(PostgreSQL)의 PostgresSaver로 교체하기 용이하도록 분리했습니다.
    """
    return MemorySaver()
