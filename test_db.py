import asyncio
from app.services.pdf_ingestion_service import get_all_registry_documents
from dotenv import load_dotenv

load_dotenv()
docs = get_all_registry_documents()
for d in docs[:5]:
    print(d)
