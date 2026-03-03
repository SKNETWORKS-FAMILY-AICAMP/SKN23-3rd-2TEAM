import os
import requests
import asyncio
from dotenv import load_dotenv

load_dotenv()
from infrastructure.aws.s3_utils import S3Client

def test_delete_endpoint():
    s3 = S3Client()
    
    # 1. First, check what's in S3
    bucket = s3.bucket_name
    print(f"Bucket: {bucket}")
    
    # Let's clean up existing orphaned files to give user a clean slate
    print("Listing existing ingested_docs...")
    res = s3.client.list_objects_v2(Bucket=bucket, Prefix="ingested_docs/")
    contents = res.get('Contents', [])
    if contents:
        keys_to_delete = [obj['Key'] for obj in contents]
        print(f"Found {len(keys_to_delete)} orphaned files in S3. Deleting them...")
        del_response = s3.delete_files(keys_to_delete)
        print("Cleanup response:", del_response)
        
test_delete_endpoint()
