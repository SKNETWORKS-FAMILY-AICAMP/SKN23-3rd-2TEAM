from __future__ import annotations

import boto3


class BedrockClient:
    def __init__(self, region: str):
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def invoke_model(self, model_id: str, body: str) -> dict:
        response = self.client.invoke_model(modelId=model_id, body=body)
        return response
