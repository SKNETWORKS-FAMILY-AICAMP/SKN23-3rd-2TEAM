import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

class S3Client:
    def __init__(self):
        self.bucket_name = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")
        
        # Initialize boto3 client. It will automatically read credentials from environment
        # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_REGION
        self.client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )

    def upload_file_bytes(self, file_bytes: bytes, s3_key: str, content_type: str = "application/octet-stream") -> str:
        """Uploads a byte array to an S3 bucket and returns the s3 key."""
        if not self.bucket_name:
            raise ValueError("AWS_S3_BUCKET environment variable is not set")
            
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_bytes,
                ContentType=content_type
            )
            return s3_key
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise RuntimeError(f"AWS credentials not found or incomplete: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to upload to S3: {e}")

    def delete_files(self, s3_keys: list[str]) -> dict:
        """Deletes multiple files from S3 given their keys."""
        if not self.bucket_name:
            raise ValueError("AWS_S3_BUCKET environment variable is not set")
            
        if not s3_keys:
            return {"Deleted": [], "Errors": []}
            
        objects_to_delete = [{'Key': key} for key in s3_keys]
        try:
            response = self.client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': objects_to_delete, 'Quiet': False}
            )
            return {
                "Deleted": response.get("Deleted", []),
                "Errors": response.get("Errors", [])
            }
        except Exception as e:
            raise RuntimeError(f"Failed to delete from S3: {e}")
