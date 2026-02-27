import os
from dotenv import load_dotenv

load_dotenv()

from infrastructure.aws.s3_utils import S3Client

s3 = S3Client()
print(f"Bucket: {s3.bucket_name}")
try:
    print("Testing list")
    response = s3.client.list_objects_v2(Bucket=s3.bucket_name, MaxKeys=5)
    contents = response.get('Contents', [])
    if contents:
        first_key = contents[0]['Key']
        print(f"Attempting to delete {first_key}")
        res = s3.delete_files([first_key])
        print("Delete Response:", res)
    else:
        print("No files to delete")
        
except Exception as e:
    print(f"Error: {e}")
