from __future__ import annotations

import boto3


class S3Client:
    def __init__(self, region: str):
        self.client = boto3.client("s3", region_name=region)

    def list_objects(self, bucket: str, prefix: str = "") -> list[dict]:
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return response.get("Contents", [])
