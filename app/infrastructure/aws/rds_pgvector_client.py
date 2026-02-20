from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class RdsPgvectorClient:
    def __init__(self, dsn: str):
        self.engine: Engine = create_engine(dsn)

    def connect(self):
        return self.engine.connect()
