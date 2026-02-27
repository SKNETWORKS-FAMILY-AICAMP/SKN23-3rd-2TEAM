import psycopg2
from app.core.database import get_connection_kwargs, open_optional_ssh_tunnel

with open_optional_ssh_tunnel() as tunnel:
    conn_args = get_connection_kwargs()
    if tunnel:
        conn_args["host"] = tunnel["host"]
        conn_args["port"] = tunnel["port"]
    
    with psycopg2.connect(**conn_args) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'langchain_pg_embedding' AND column_name = 'embedding';
            """)
            print(cur.fetchall())
