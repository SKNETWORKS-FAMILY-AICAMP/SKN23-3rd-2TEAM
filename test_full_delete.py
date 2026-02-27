import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from app.services.pdf_ingestion_service import get_all_registry_documents, delete_registry_documents_by_sources
from infrastructure.aws.s3_utils import S3Client

docs = get_all_registry_documents()
print("Registry docs:", docs)

# Try deleting the first one?
if docs:
    first_source = docs[0]["source_key"]
    print("Trying to delete:", first_source)
    res = delete_registry_documents_by_sources([first_source])
    print("Delete Registry Result:", res)
    s3_keys = res.get("s3_keys_to_delete", [])
    if s3_keys:
        print("S3 keys to delete:", s3_keys)
        s3 = S3Client()
        del_res = s3.delete_files(s3_keys)
        print("S3 delete response:", del_res)
