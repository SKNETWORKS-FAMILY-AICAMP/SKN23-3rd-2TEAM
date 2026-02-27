import os
from dotenv import load_dotenv

load_dotenv()

from infrastructure.aws.s3_utils import S3Client

s3 = S3Client()
print(f"Bucket: {s3.bucket_name}")
try:
    print("Testing list")
    response = s3.client.list_objects_v2(Bucket=s3.bucket_name, Prefix="ingested_docs/")
    contents = response.get('Contents', [])
    for obj in contents[-20:]:  # Print last 20
        print(obj['Key'])
    print(f"Total found with prefix ingested_docs/: {len(contents)}")
except Exception as e:
    print(f"Error: {e}")
