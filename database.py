"""Conexión Singleton (thread-safe) a MongoDB."""
from __future__ import annotations

import threading
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database

from config import get_settings


class MongoConnection:
    _client: Optional[MongoClient] = None
    _lock = threading.Lock()

    @classmethod
    def get_client(cls) -> MongoClient:
        if cls._client is None:
            with cls._lock:
                if cls._client is None:
                    settings = get_settings()
                    client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
                    client.admin.command("ping")  # Falla rápido si Mongo no está disponible
                    cls._client = client
        return cls._client

    @classmethod
    def get_db(cls) -> Database:
        return cls.get_client()[get_settings().mongo_db]

    @classmethod
    def close(cls) -> None:
        with cls._lock:
            if cls._client is not None:
                cls._client.close()
                cls._client = None