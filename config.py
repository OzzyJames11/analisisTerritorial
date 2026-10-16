"""Configuración centralizada. Única fuente de verdad para variables de entorno."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(f"Variable de entorno obligatoria no definida: {name}")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} debe ser un entero, se recibió: {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} debe ser >= 1, se recibió: {value}")
    return value


@dataclass(frozen=True)
class Settings:
    mongo_uri: str
    mongo_db: str
    min_cluster_size: int
    excel_path: Path
    output_dir: Path
    coll_academic: str = "oferta_academica"
    coll_jobs: str = "gold_jobs"
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16
    embedding_max_seq_length: int = 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    excel_path = Path(os.getenv("EXCEL_PATH", "data/oferta_academica.xlsx"))
    if not excel_path.is_absolute():
        excel_path = BASE_DIR / excel_path

    return Settings(
        mongo_uri=_required("MONGO_URI"),
        mongo_db=os.getenv("MONGO_DB", "TalentMine").strip() or "TalentMine",
        min_cluster_size=_positive_int("MIN_CLUSTER_SIZE", 1),
        excel_path=excel_path,
        output_dir=BASE_DIR / "outputs",
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
        embedding_batch_size=_positive_int("EMBEDDING_BATCH_SIZE", 16),
        embedding_max_seq_length=_positive_int("EMBEDDING_MAX_SEQ_LENGTH", 1024),
    )