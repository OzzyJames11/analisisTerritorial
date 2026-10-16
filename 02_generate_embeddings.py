"""
Paso 2 (ETL 2): Generación de embeddings para `oferta_academica`.

Concatena nombre + perfil_egreso + perfil_profesional, genera el vector de 1024
dimensiones con BAAI/bge-m3 (CPU estricto) y lo guarda en `embedding_bge`.

Por defecto procesa solo los documentos sin embedding válido (reanudable).
Uso:
    python 02_generate_embeddings.py          # solo pendientes / inválidos
    python 02_generate_embeddings.py --force  # recalcula todos
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Iterator, Sequence

from pymongo import UpdateOne
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import get_settings
from database import MongoConnection

TEXT_FIELDS = ("nombre", "perfil_egreso", "perfil_profesional")
EMBEDDING_FIELD = "embedding_bge"
EMBEDDING_DIM = 1024


# ----------------------------------------------------------------------------
# Lógica pura (sin I/O)
# ----------------------------------------------------------------------------
def _as_text(value) -> str:
    """Devuelve el texto limpio; cualquier valor no-string (None, NaN) cuenta como vacío."""
    return value.strip() if isinstance(value, str) else ""


def build_text(doc: dict) -> str:
    """Concatena los campos de texto no vacíos, separados por salto de línea."""
    parts = (_as_text(doc.get(field)) for field in TEXT_FIELDS)
    return "\n".join(p for p in parts if p)


def chunked(items: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def load_model(name: str, max_seq_length: int) -> SentenceTransformer:
    model = SentenceTransformer(name, device="cpu")  # CPU estricto: sin MPS ni CUDA
    model.max_seq_length = max_seq_length
    dim = model.get_sentence_embedding_dimension()
    if dim != EMBEDDING_DIM:
        raise ValueError(f"El modelo {name} produce {dim} dimensiones, se esperaban {EMBEDDING_DIM}")
    return model


# ----------------------------------------------------------------------------
# Acceso a datos
# ----------------------------------------------------------------------------
def pending_filter(force: bool) -> dict:
    """Filtro de documentos a procesar: todos (--force) o los que no tienen vector válido."""
    if force:
        return {}
    return {
        "$or": [
            {EMBEDDING_FIELD: {"$exists": False}},
            {EMBEDDING_FIELD: None},
            {EMBEDDING_FIELD: {"$not": {"$size": EMBEDDING_DIM}}},
        ]
    }


def fetch_documents(collection: Collection, query: dict) -> list[dict]:
    projection = {field: 1 for field in TEXT_FIELDS}
    return list(collection.find(query, projection))


def save_embeddings(collection: Collection, ids: Sequence, vectors) -> int:
    operations = [
        UpdateOne({"_id": _id}, {"$set": {EMBEDDING_FIELD: vector.astype(float).tolist()}})
        for _id, vector in zip(ids, vectors)
    ]
    result = collection.bulk_write(operations, ordered=False)
    return result.matched_count


def count_valid_embeddings(collection: Collection) -> int:
    return collection.count_documents({EMBEDDING_FIELD: {"$size": EMBEDDING_DIM}})


# ----------------------------------------------------------------------------
# Orquestación
# ----------------------------------------------------------------------------
def print_summary(total: int, to_process: int, updated: int, without_text: int,
                  valid: int, elapsed: float) -> None:
    line = "=" * 58
    print(line)
    print(" RESUMEN DE EMBEDDINGS - oferta_academica")
    print(line)
    print(f" Documentos en la colección ........ {total}")
    print(f" Documentos a procesar ............. {to_process}")
    print(f" Actualizados con embedding_bge .... {updated}")
    print(f" Omitidos por texto vacío .......... {without_text}")
    print(f" Con embedding válido ({EMBEDDING_DIM} dims) .. {valid}")
    print(f" Tiempo total ...................... {elapsed:.1f} s")
    print(line)
    print(" ESTADO:", "OK" if valid == total else "INCOMPLETO - revisar documentos sin embedding")
    print(line)


def run(force: bool) -> int:
    settings = get_settings()
    collection = MongoConnection.get_db()[settings.coll_academic]

    total = collection.count_documents({})
    if total == 0:
        print(f"[ERROR] La colección {settings.coll_academic} está vacía. Ejecuta primero 01_ingest_excel.py",
              file=sys.stderr)
        return 2

    documents = fetch_documents(collection, pending_filter(force))
    if not documents:
        print(f"Nada que procesar: los {total} documentos ya tienen embedding válido.")
        print_summary(total, 0, 0, 0, count_valid_embeddings(collection), 0.0)
        return 0

    start = time.perf_counter()
    print(f"Cargando modelo {settings.embedding_model} en CPU...")
    model = load_model(settings.embedding_model, settings.embedding_max_seq_length)

    updated = 0
    without_text = 0
    with tqdm(total=len(documents), desc="Embeddings", unit="doc") as progress:
        for batch in chunked(documents, settings.embedding_batch_size):
            texts = [build_text(doc) for doc in batch]
            valid_pairs = [(doc["_id"], text) for doc, text in zip(batch, texts) if text]
            without_text += len(batch) - len(valid_pairs)

            if valid_pairs:
                ids, valid_texts = zip(*valid_pairs)
                vectors = model.encode(
                    list(valid_texts),
                    batch_size=settings.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                updated += save_embeddings(collection, ids, vectors)
            progress.update(len(batch))

    valid = count_valid_embeddings(collection)
    print_summary(total, len(documents), updated, without_text, valid, time.perf_counter() - start)
    return 0 if valid == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera embedding_bge para oferta_academica")
    parser.add_argument("--force", action="store_true", help="Recalcula todos los embeddings")
    args = parser.parse_args()

    try:
        return run(force=args.force)
    except (ValueError, EnvironmentError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except PyMongoError as exc:
        print(f"[ERROR MongoDB] {exc}", file=sys.stderr)
        return 3
    finally:
        MongoConnection.close()


if __name__ == "__main__":
    sys.exit(main())