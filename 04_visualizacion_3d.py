"""
Fase 4: Visualización 3D de un clúster laboral (UMAP + Plotly).

Dada una ciudad y una carrera (`carrera_epn_oficial`), proyecta a 3D:
  - las ofertas laborales de gold_jobs del clúster (mismo criterio que la Fase 3),
  - el centroide del clúster,
  - la(s) oferta(s) académica(s) de esa carrera en esa ciudad,
y guarda un HTML interactivo en outputs/cluster_3d.html.

Uso:
    python 04_visualizacion_3d.py --ciudad Quito --carrera "Ingenieria En Software"
    python 04_visualizacion_3d.py            # pide los datos por terminal (o usa las variables de abajo)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import webbrowser
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from scipy.spatial.distance import cosine

from config import get_settings
from database import MongoConnection
from services.geo import build_city_index, normalize_key, resolve_city

# Reutiliza TRADUCTOR_GOLD, to_vector, etc. de la Fase 3 (DRY). El nombre empieza por
# dígito, por lo que se importa con importlib; el guard __main__ evita ejecutar el motor.
engine = importlib.import_module("03_analytics_engine")

# ----------------------------------------------------------------------------
# Parámetros editables
# ----------------------------------------------------------------------------
CIUDAD_OBJETIVO = ""    # p. ej. "Quito". Si queda vacío: --ciudad o input()
CARRERA_OBJETIVO = ""   # p. ej. "Ingenieria En Software". Si queda vacío: --carrera o input()

# Qué hacer si varias universidades ofrecen la carrera en la ciudad:
#   "todas"    -> un punto por programa (coincide con las entradas de output1_distancias.json)
#   "promedio" -> un único punto con el promedio de los vectores académicos
#   "primera"  -> solo el primer programa encontrado
MODO_ACADEMICO = "todas"

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
MIN_JOBS_FOR_UMAP = 3

OUTPUT_FILE = "cluster_3d.html"

# Campos de gold_jobs usados para el tooltip. AJÚSTALOS al esquema real de tu colección:
# se usa el primero que exista y sea texto.
JOB_TITLE_FIELDS = ("titulo", "title", "cargo", "puesto", "job_title", "nombre")
JOB_COMPANY_FIELDS = ("empresa", "company", "compania")

T_JOB, T_CENTROID, T_ACADEMIC = "Oferta Laboral", "Centroide", "Oferta Académica"
COLORS = {T_JOB: "#4C78A8", T_CENTROID: "#E45756", T_ACADEMIC: "#2CA02C"}
SYMBOLS = {T_JOB: "circle", T_CENTROID: "diamond", T_ACADEMIC: "square"}
SIZES = {T_JOB: 2, T_CENTROID: 18, T_ACADEMIC: 18}


# ----------------------------------------------------------------------------
# Utilidades puras
# ----------------------------------------------------------------------------
def first_text(doc: dict, fields: tuple) -> str:
    for field in fields:
        value = doc.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def shorten(text: str, limit: int = 110) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def resolve_inputs(args: argparse.Namespace) -> tuple:
    ciudad = (args.ciudad or CIUDAD_OBJETIVO or input("Ciudad objetivo (ej. Quito): ")).strip()
    carrera = (args.carrera or CARRERA_OBJETIVO or input("Carrera objetivo (ej. Ingenieria En Software): ")).strip()
    if not ciudad or not carrera:
        raise ValueError("Debes indicar ciudad y carrera.")
    return ciudad, carrera


# ----------------------------------------------------------------------------
# Extracción de datos
# ----------------------------------------------------------------------------
def load_academic(academic: Collection, ciudad: str, carrera: str) -> list:
    """Programas académicos de la carrera en la ciudad (comparación normalizada) con embedding válido."""
    city_key, career_key = normalize_key(ciudad), normalize_key(carrera)
    projection = {"ciudad": 1, "nombre": 1, "universidad": 1, "carrera_epn_oficial": 1, engine.EMBEDDING_FIELD: 1}

    rows = []
    for doc in academic.find({}, projection):
        if normalize_key(engine.clean_text(doc.get("ciudad"))) != city_key:
            continue
        if normalize_key(engine.clean_text(doc.get("carrera_epn_oficial"))) != career_key:
            continue
        vector = engine.to_vector(doc.get(engine.EMBEDDING_FIELD))
        if vector is None:
            continue
        rows.append({
            "vector": vector,
            "universidad": engine.clean_text(doc.get("universidad")) or "(sin universidad)",
            "nombre": engine.clean_text(doc.get("nombre")) or "(sin nombre)",
        })
    return rows


def select_academic(rows: list, mode: str) -> list:
    """Aplica MODO_ACADEMICO y devuelve [{'vector', 'nombre', 'detalle'}]."""
    if mode == "todas":
        chosen = rows
    elif mode == "primera":
        chosen = rows[:1]
    elif mode == "promedio":
        mean_vector = np.mean(np.vstack([r["vector"] for r in rows]), axis=0)
        universities = ", ".join(sorted({r["universidad"] for r in rows}))
        return [{"vector": mean_vector, "nombre": f"Promedio de {len(rows)} programa(s)",
                 "detalle": shorten(universities)}]
    else:
        raise ValueError(f"MODO_ACADEMICO inválido: {mode!r} (usa 'todas', 'promedio' o 'primera')")

    return [{"vector": r["vector"], "nombre": r["nombre"], "detalle": r["universidad"]} for r in chosen]


def load_jobs(jobs: Collection, spellings: list, labels: tuple) -> tuple:
    """Ofertas del clúster (misma consulta que la Fase 3). Devuelve (vectores, metadatos, corruptas)."""
    query = {"ciudad": {"$in": spellings}, "carreras_relacionadas": {"$in": list(labels)}}
    projection = {field: 1 for field in JOB_TITLE_FIELDS + JOB_COMPANY_FIELDS}
    projection.update({engine.EMBEDDING_FIELD: 1, "carreras_relacionadas": 1})

    vectors, meta, corrupt = [], [], 0
    for doc in jobs.find(query, projection):
        vector = engine.to_vector(doc.get(engine.EMBEDDING_FIELD))
        if vector is None:
            corrupt += 1
            continue
        title = first_text(doc, JOB_TITLE_FIELDS) or f"Oferta …{str(doc['_id'])[-6:]}"
        company = first_text(doc, JOB_COMPANY_FIELDS)
        careers = doc.get("carreras_relacionadas")
        careers_text = ", ".join(c for c in careers if isinstance(c, str)) if isinstance(careers, list) else ""
        detail = " | ".join(part for part in (company, careers_text) if part)
        vectors.append(vector)
        meta.append({"nombre": shorten(title, 90), "detalle": shorten(detail)})
    return vectors, meta, corrupt


# ----------------------------------------------------------------------------
# Reducción de dimensionalidad y gráfico
# ----------------------------------------------------------------------------
def reduce_to_3d(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    reducer = umap.UMAP(
        n_components=3,
        metric="cosine",
        n_neighbors=max(2, min(UMAP_N_NEIGHBORS, n - 1)),
        min_dist=UMAP_MIN_DIST,
        init="spectral" if n > 10 else "random",  # spectral es inestable con muy pocos puntos
        random_state=UMAP_RANDOM_STATE,
        n_jobs=1,  # con random_state fijo UMAP es single-thread; se explicita para evitar el aviso
    )
    return reducer.fit_transform(matrix)


def build_dataframe(coords: np.ndarray, job_meta: list, academic: list, distances: list) -> pd.DataFrame:
    n_jobs = len(job_meta)
    rows = [{"Tipo": T_JOB, "Nombre": m["nombre"], "Detalle": m["detalle"]} for m in job_meta]
    rows.append({"Tipo": T_CENTROID, "Nombre": f"Centroide ({n_jobs} ofertas)", "Detalle": "Media de los embeddings"})
    for item, dist in zip(academic, distances):
        rows.append({"Tipo": T_ACADEMIC, "Nombre": item["nombre"],
                     "Detalle": f"{item['detalle']} | distancia coseno al centroide: {dist:.4f}"})

    df = pd.DataFrame(rows)
    df[["x", "y", "z"]] = coords
    df["Tamano"] = df["Tipo"].map(SIZES)
    return df


def build_figure(df: pd.DataFrame, ciudad: str, carrera: str, n_jobs: int, distances: list) -> go.Figure:
    distance_text = ", ".join(f"{d:.4f}" for d in distances)
    fig = px.scatter_3d(
        df, x="x", y="y", z="z",
        color="Tipo", symbol="Tipo", size="Tamano", size_max=18,
        hover_name="Nombre",
        hover_data={"Detalle": True, "Tipo": False, "Tamano": False, "x": False, "y": False, "z": False},
        color_discrete_map=COLORS, symbol_map=SYMBOLS,
        category_orders={"Tipo": [T_JOB, T_CENTROID, T_ACADEMIC]},
        title=(f"{carrera} · {ciudad} — {n_jobs} ofertas laborales<br>"
               f"<sup>Distancia coseno carrera→centroide: {distance_text} "
               f"(la distancia visual en UMAP es solo ilustrativa)</sup>"),
        template="plotly_white",
    )
    fig.update_traces(marker=dict(opacity=0.55), selector=dict(name=T_JOB))
    for special in (T_CENTROID, T_ACADEMIC):
        fig.update_traces(marker=dict(opacity=1.0, line=dict(width=2, color="black")), selector=dict(name=special))

    # Segmento academia -> centroide, con la distancia coseno real en el tooltip
    centroid = df[df["Tipo"] == T_CENTROID].iloc[0]
    for (_, point), dist in zip(df[df["Tipo"] == T_ACADEMIC].iterrows(), distances):
        fig.add_trace(go.Scatter3d(
            x=[point["x"], centroid["x"]], y=[point["y"], centroid["y"]], z=[point["z"], centroid["z"]],
            mode="lines", line=dict(color="gray", width=4),
            hoverinfo="text", text=f"Distancia coseno real: {dist:.4f}", showlegend=False,
        ))

    fig.update_layout(
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend_title_text="Tipo", margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig


# ----------------------------------------------------------------------------
# Orquestación
# ----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    db = MongoConnection.get_db()
    academic_coll, jobs_coll = db[settings.coll_academic], db[settings.coll_jobs]

    ciudad, carrera = resolve_inputs(args)

    translator_index = engine.build_translator_index(engine.TRADUCTOR_GOLD)
    labels = engine.translate_career(carrera, translator_index)
    if labels is None:
        raise ValueError(f"{carrera!r} no está en TRADUCTOR_GOLD. Opciones: {sorted(engine.TRADUCTOR_GOLD)}")

    city_index = build_city_index(jobs_coll.distinct("ciudad"))
    spellings = resolve_city(ciudad, city_index)
    if not spellings:
        raise ValueError(f"La ciudad {ciudad!r} no tiene equivalente en gold_jobs. "
                         f"Ciudades disponibles: {sorted(v[0] for v in city_index.values())}")

    academic_rows = load_academic(academic_coll, ciudad, carrera)
    if not academic_rows:
        raise ValueError(f"No hay oferta académica con embedding para {carrera!r} en {ciudad!r}.")
    academic = select_academic(academic_rows, MODO_ACADEMICO)

    job_vectors, job_meta, corrupt = load_jobs(jobs_coll, spellings, labels)
    if len(job_vectors) < MIN_JOBS_FOR_UMAP:
        raise ValueError(f"Clúster demasiado pequeño para UMAP: {len(job_vectors)} ofertas válidas "
                         f"(mínimo {MIN_JOBS_FOR_UMAP}).")
    if len(job_vectors) < settings.min_cluster_size:
        print(f"[ADVERTENCIA] El clúster tiene {len(job_vectors)} ofertas (< MIN_CLUSTER_SIZE="
              f"{settings.min_cluster_size}): en la Fase 3 esta carrera sería huérfana.", file=sys.stderr)

    centroid = engine.compute_centroid(job_vectors)
    distances = [float(cosine(item["vector"], centroid)) for item in academic]

    # Orden: ofertas laborales + centroide + academia (coincide con build_dataframe)
    matrix = np.vstack(job_vectors + [centroid] + [item["vector"] for item in academic])
    print(f"Ejecutando UMAP sobre {matrix.shape[0]} puntos de {matrix.shape[1]} dimensiones...")
    coords = reduce_to_3d(matrix)

    df = build_dataframe(coords, job_meta, academic, distances)
    fig = build_figure(df, ciudad, carrera, len(job_vectors), distances)

    output_path = settings.output_dir / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)  # JS embebido: funciona sin internet

    print("=" * 62)
    print(f" Carrera ............ {carrera}")
    print(f" Ciudad ............. {ciudad} (gold_jobs: {', '.join(spellings)})")
    print(f" Etiquetas Gold ..... {', '.join(labels)}")
    print(f" Ofertas válidas .... {len(job_vectors)} (descartadas por embedding: {corrupt})")
    print(f" Puntos académicos .. {len(academic)} (modo: {MODO_ACADEMICO})")
    for item, dist in zip(academic, distances):
        print(f"     - {item['detalle']}: distancia coseno {dist:.4f}")
    print(f" Archivo generado ... {output_path}")
    print("=" * 62)

    try:
        webbrowser.open(output_path.resolve().as_uri())
    except Exception:  # noqa: BLE001 - abrir el navegador es opcional
        print("No se pudo abrir el navegador automáticamente; abre el archivo manualmente.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualización 3D de un clúster laboral")
    parser.add_argument("--ciudad", help="Ciudad objetivo (ej. Quito)")
    parser.add_argument("--carrera", help="carrera_epn_oficial (ej. 'Ingenieria En Software')")
    args = parser.parse_args()

    try:
        return run(args)
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