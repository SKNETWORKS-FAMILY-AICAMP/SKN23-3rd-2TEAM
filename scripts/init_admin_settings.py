import os
import sys
from pathlib import Path

# Add project root to sys.path so 'app' can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

import psycopg2
from app.core.database import init_admin_settings_table, open_optional_ssh_tunnel, get_connection_kwargs

def alter_chat_logs_table():
    print("🔄 Altering chat_logs table to add evaluation columns...")
    try:
        with open_optional_ssh_tunnel() as tunnel:
            conn_args = get_connection_kwargs()
            if tunnel:
                conn_args["host"] = tunnel["host"]
                conn_args["port"] = tunnel["port"]
            
            with psycopg2.connect(**conn_args) as conn:
                with conn.cursor() as cur:
                    # Add generation_score (int)
                    cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS generation_score INT")
                    # Add retrieval_total_chunks (int)
                    cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS retrieval_total_chunks INT")
                    # Add retrieval_relevant_chunks (int)
                    cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS retrieval_relevant_chunks INT")
                    # Add retrieval_is_answerable (boolean)
                    cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS retrieval_is_answerable BOOLEAN")
                    # Add eval_reason (text)
                    cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS eval_reason TEXT")
                    # Add context (text) for UI display 
                    cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS context TEXT")
                    # Add models (varchar)
                    cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS generation_model VARCHAR(100)")
                    cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS evaluation_model VARCHAR(100)")
                conn.commit()
        print("✅ chat_logs altered successfully.")
    except Exception as e:
        print(f"❌ Altering chat_logs failed: {e}")

if __name__ == "__main__":
    print("🚀 Initializing admin settings table...")
    init_admin_settings_table()
    print("✅ Admin settings initialized.")
    alter_chat_logs_table()
