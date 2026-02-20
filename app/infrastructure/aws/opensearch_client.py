from __future__ import annotations

from opensearchpy import OpenSearch


class OpenSearchClient:
    def __init__(self, host: str, port: int, use_ssl: bool = True):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            use_ssl=use_ssl,
            verify_certs=use_ssl,
        )

    def info(self) -> dict:
        return self.client.info()
