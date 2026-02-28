from app.core.database import get_chat_logs
logs = get_chat_logs(limit=1000)
print("Logs:", len(logs))
