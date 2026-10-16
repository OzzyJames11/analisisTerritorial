"""
Paso 3 y 4: Motor analítico y generación de entregables.

Por cada carrera de `oferta_academica`:
  1. Traduce `carrera_epn_oficial` a las etiquetas de gold_jobs (TRADUCTOR_GOLD).
  2. Resuelve la ciudad contra gold_jobs (normalización geográfica).
  3. Arma el clúster laboral: misma ciudad + carreras_relacionadas contiene AL MENOS UNA
     de las etiquetas traducidas ($in) + embedding_bge válido (1024 floats).
     Una oferta que tenga varias etiquetas del mismo mapeo cuenta una sola vez.
  4. Si el clúster tiene >= MIN_CLUSTER_SIZE ofertas, calcula el centroide (media, eje 0).
  5. Calcula la distancia coseno (scipy) entre la carrera y el centroide.

Salidas en outputs/:
  - output1_distancias.json : {ciudad: [{carrera, distanciaCarreraCentroide}, ...]}
  - output2_indice.json     : [{ciudad, promedio}, ...] ascendente por promedio
  - huerfanas_analitica.json: carreras sin distancia y su motivo (trazabilidad)

Uso: python 03_analytics_engine.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from scipy.spatial.distance import cosine
from tqdm import tqdm

from config import get_settings
from database import MongoConnection
from services.geo import build_city_index, normalize_key, resolve_city

# Mapeo de oferta_academica -> etiquetas en gold_jobs
TRADUCTOR_GOLD = {
    "Licenciatura en Administracion De Empresas": ["Administración de Empresas"],
    "Ingenieria Agroindustria": ["Agroindustria"],
    "Ingenieria Ambiental": ["Ingeniería Ambiental"],
    # Esta carrera académica mapea a dos posibles etiquetas laborales
    "Ciencias De Datos E Inteligencia Artificial": ["Ciencia de Datos", "Inteligencia Artificial"],
    "Ingenieria En Ciencias De La Computacion": ["Computación"],
    "Economia": ["Economía"],
    "Ingenieria En Electricidad": ["Electricidad"],
    "Ingenieria En Electronica Y Automatizacion": ["Electrónica y Automatización"],
    "Fisica": ["Física"],
    "Ingenieria En Geologia": ["Geología"],
    "Ingenieria De La Produccion": ["Ingeniería de la Producción"],
    "Matematica": ["Matemática"],
    "Matematica Aplicada": ["Matemática Aplicada"],
    "Ingenieria En Materiales": ["Materiales"],
    "Ingenieria En Mecanica": ["Mecánica"],
    "Ingenieria En Mecatronica": ["Mecatrónica"],
    "Ingenieria En Petroleos": ["Petróleos"],
    "Ingenieria Quimica": ["Ingeniería Química"],
    "Ingenieria En Sistemas De Informacion": ["Sistemas de Información"],
    "Ingenieria En Software": ["Software"],
    "Ingenieria En Tecnologias De La Informacion": ["Tecnologías de la Información"],
    "Ingenieria En Telecomunicaciones": ["Telecomunicaciones"],
    "Ingenieria Civil": ["Ingeniería Civil"],
    # Mapeo por similitud de área
    "Seguridad De Redes De Informacion": ["Tecnologías de la Información"],
}

EMBEDDING_FIELD = "embedding_bge"
EMBEDDING_DIM = 1024
ROUND_DECIMALS = 6

OUTPUT1 = "output1_distancias.json"
OUTPUT2 = "output2_indice.json"
OUTPUT_ORPHANS = "huerfanas_analitica.json"

# Motivos de carrera huérfana
R_NO_CARRERA = "sin_carrera_epn_oficial"
R_NO_CITY = "ciudad_vacia"
R_NO_TRANSLATION = "carrera_sin_traduccion_en_traductor"
R_NO_EMBEDDING = "embedding_academico_invalido"
R_CITY_NOT_IN_JOBS = "ciudad_sin_match_en_gold_jobs"
R_SMALL_CLUSTER = "cluster_menor_a_min_cluster_size"
R_NON_FINITE = "distancia_no_finita"


@dataclass(frozen=True)
class ClusterResult:
    size: int
    corrupt: int
    centroid: Optional[np.ndarray]


# ----------------------------------------------------------------------------
# Lógica pura
# ----------------------------------------------------------------------------
def clean_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_translator_index(translator: dict) -> dict:
    """Índice {clave_normalizada: tupla de etiquetas Gold}. Valida el diccionario al arrancar."""
    index: dict = {}
    for key, labels in translator.items():
        if not labels or not all(isinstance(label, str) and label.strip() for label in labels):
            raise ValueError(f"TRADUCTOR_GOLD: la clave {key!r} debe tener al menos una etiqueta de texto")
        norm = normalize_key(key)
        if norm in index:
            raise ValueError(f"TRADUCTOR_GOLD: clave duplicada tras normalizar: {key!r}")
        index[norm] = tuple(labels)
    return index


def translate_career(carrera: str, translator_index: dict) -> Optional[tuple]:
    """Etiquetas Gold para `carrera_epn_oficial`; None si no está en el traductor."""
    return translator_index.get(normalize_key(carrera))


def to_vector(value) -> Optional[np.ndarray]:
    """Convierte un embedding de Mongo a ndarray float64; None si es inválido o corrupto."""
    if not isinstance(value, list) or len(value) != EMBEDDING_DIM:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
        return None
    vector = np.asarray(value, dtype=np.float64)
    return vector if np.isfinite(vector).all() else None


def compute_centroid(vectors: list) -> np.ndarray:
    return np.mean(np.vstack(vectors), axis=0)


# ----------------------------------------------------------------------------
# Acceso a datos
# ----------------------------------------------------------------------------
def missing_gold_labels(jobs: Collection, translator: dict) -> list:
    """Etiquetas del traductor que NO existen en ningún carreras_relacionadas de gold_jobs."""
    existing = set(jobs.distinct("carreras_relacionadas"))
    wanted = {label for labels in translator.values() for label in labels}
    return sorted(wanted - existing)


def build_cluster(jobs: Collection, spellings: list, labels: tuple, min_size: int) -> ClusterResult:
    query = {"ciudad": {"$in": spellings}, "carreras_relacionadas": {"$in": list(labels)}}
    vectors, corrupt = [], 0
    for doc in jobs.find(query, {EMBEDDING_FIELD: 1}):
        vector = to_vector(doc.get(EMBEDDING_FIELD))
        if vector is None:
            corrupt += 1
        else:
            vectors.append(vector)

    centroid = compute_centroid(vectors) if vectors and len(vectors) >= min_size else None
    return ClusterResult(size=len(vectors), corrupt=corrupt, centroid=centroid)


# ----------------------------------------------------------------------------
# Salida
# ----------------------------------------------------------------------------
def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def build_outputs(distances_by_city: dict) -> tuple:
    output1 = {
        city: sorted(items, key=lambda item: item["distanciaCarreraCentroide"])
        for city, items in sorted(distances_by_city.items())
    }
    output2 = sorted(
        (
            {
                "ciudad": city,
                "promedio": round(
                    float(np.mean([i["distanciaCarreraCentroide"] for i in items])), ROUND_DECIMALS
                ),
            }
            for city, items in distances_by_city.items()
        ),
        key=lambda row: (row["promedio"], row["ciudad"]),
    )
    return output1, output2


# ----------------------------------------------------------------------------
# Orquestación
# ----------------------------------------------------------------------------
def run() -> int:
    settings = get_settings()
    db = MongoConnection.get_db()
    academic = db[settings.coll_academic]
    jobs = db[settings.coll_jobs]
    min_size = settings.min_cluster_size

    programs = list(academic.find({}, {
        "ciudad": 1, "nombre": 1, "universidad": 1, "carrera_epn_oficial": 1, EMBEDDING_FIELD: 1,
    }))
    if not programs:
        print(f"[ERROR] La colección {settings.coll_academic} está vacía.", file=sys.stderr)
        return 2

    start = time.perf_counter()
    translator_index = build_translator_index(TRADUCTOR_GOLD)
    city_index = build_city_index(jobs.distinct("ciudad"))
    jobs_without_city = jobs.count_documents({"ciudad": {"$in": [None]}})

    # Diagnóstico previo: etiquetas del traductor inexistentes en gold_jobs (typos / diccionario desfasado)
    absent_labels = missing_gold_labels(jobs, TRADUCTOR_GOLD)
    for label in absent_labels:
        print(f"[ADVERTENCIA] La etiqueta {label!r} del traductor no existe en gold_jobs.carreras_relacionadas",
              file=sys.stderr)

    cache: dict = {}
    distances_by_city: dict = {}
    orphans: list = []
    reasons: Counter = Counter()
    unresolved_cities: Counter = Counter()
    untranslated: Counter = Counter()

    def register_orphan(doc: dict, reason: str, cluster_size: Optional[int] = None) -> None:
        reasons[reason] += 1
        orphans.append({
            "id": str(doc["_id"]),
            "ciudad": clean_text(doc.get("ciudad")) or None,
            "universidad": clean_text(doc.get("universidad")) or None,
            "nombre": clean_text(doc.get("nombre")) or None,
            "carrera_epn_oficial": clean_text(doc.get("carrera_epn_oficial")) or None,
            "motivo": reason,
            "tamanoCluster": cluster_size,
        })

    for doc in tqdm(programs, desc="Analizando carreras", unit="carrera"):
        city = clean_text(doc.get("ciudad"))
        carrera = clean_text(doc.get("carrera_epn_oficial"))

        if not carrera:
            register_orphan(doc, R_NO_CARRERA)
            continue

        labels = translate_career(carrera, translator_index)
        if labels is None:
            if untranslated[carrera] == 0:  # una sola advertencia por carrera
                tqdm.write(f"[ADVERTENCIA] carrera_epn_oficial {carrera!r} no está en TRADUCTOR_GOLD; se omite")
            untranslated[carrera] += 1
            register_orphan(doc, R_NO_TRANSLATION)
            continue

        if not city:
            register_orphan(doc, R_NO_CITY)
            continue

        program_vector = to_vector(doc.get(EMBEDDING_FIELD))
        if program_vector is None:
            register_orphan(doc, R_NO_EMBEDDING)
            continue

        spellings = resolve_city(city, city_index)
        if not spellings:
            unresolved_cities[city] += 1
            register_orphan(doc, R_CITY_NOT_IN_JOBS)
            continue

        cache_key = (tuple(sorted(spellings)), tuple(sorted(labels)))
        if cache_key not in cache:
            cache[cache_key] = build_cluster(jobs, spellings, labels, min_size)
        cluster = cache[cache_key]

        if cluster.centroid is None:
            register_orphan(doc, R_SMALL_CLUSTER, cluster.size)
            continue

        distance = float(cosine(program_vector, cluster.centroid))
        if not np.isfinite(distance):
            register_orphan(doc, R_NON_FINITE, cluster.size)
            continue

        distances_by_city.setdefault(city, []).append({
            "universidad": clean_text(doc.get("universidad")) or "Desconocida",
            "carrera_original": clean_text(doc.get("nombre")) or "Desconocida",
            "carrera_homologada": carrera,
            "distanciaCarreraCentroide": round(distance, ROUND_DECIMALS),
        })

    # --- Entregables ---------------------------------------------------------
    if not distances_by_city:
        print("[ERROR] No se calculó ninguna distancia; no se generan los JSON de resultados.", file=sys.stderr)
        write_json(settings.output_dir / OUTPUT_ORPHANS, orphans)
        print(f" Motivos de huérfanas: {dict(reasons)}", file=sys.stderr)
        return 1

    output1, output2 = build_outputs(distances_by_city)
    write_json(settings.output_dir / OUTPUT1, output1)
    write_json(settings.output_dir / OUTPUT2, output2)
    write_json(settings.output_dir / OUTPUT_ORPHANS, orphans)

    # --- Log -------------------------------------------------------------------
    formed = [c for c in cache.values() if c.centroid is not None]
    sizes = [c.size for c in formed]
    evaluated = sum(len(items) for items in distances_by_city.values())
    duplicated = sum(1 for n in Counter(
        (city, i["carrera_homologada"]) for city, items in distances_by_city.items() for i in items
    ).values() if n > 1)

    line = "=" * 62
    print(line)
    print(" RESUMEN DEL MOTOR ANALÍTICO")
    print(line)
    print(f" MIN_CLUSTER_SIZE ................... {min_size}")
    print(f" Carreras en oferta_academica ....... {len(programs)}")
    print(f" Carreras con distancia calculada ... {evaluated}")
    print(f" Carreras huérfanas ................. {len(orphans)}")
    for reason, count in reasons.most_common():
        print(f"     - {reason}: {count}")
    print(f" Clústeres laborales formados ....... {len(formed)} de {len(cache)} evaluados")
    if sizes:
        print(f" Tamaño de clúster (mín/mediana/máx)  {min(sizes)} / {statistics.median(sizes)} / {max(sizes)}")
    print(f" Ofertas descartadas por embedding .. {sum(c.corrupt for c in cache.values())}")
    print(f" Ofertas en gold_jobs sin ciudad .... {jobs_without_city} (fuera del cruce)")
    if untranslated:
        print(f" Carreras sin traducción ............ {dict(untranslated)}")
    if absent_labels:
        print(f" Etiquetas del traductor ausentes en gold_jobs: {absent_labels}")
    if unresolved_cities:
        print(f" Ciudades sin match en gold_jobs .... {dict(unresolved_cities)}")
    if duplicated:
        print(f" Pares (ciudad, carrera) con >1 programa: {duplicated} (cada programa cuenta en el promedio)")
    print(line)
    print(" ÍNDICE (menor distancia = mayor inserción)")
    for row in output2:
        print(f"   {row['ciudad']:<14} {row['promedio']:.4f}  ({len(output1[row['ciudad']])} carreras)")
    print(line)
    for name in (OUTPUT1, OUTPUT2, OUTPUT_ORPHANS):
        print(f" Archivo generado: {settings.output_dir / name}")
    print(f" Tiempo total: {time.perf_counter() - start:.1f} s")
    print(line)
    return 0


def main() -> int:
    try:
        return run()
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