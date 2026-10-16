"""
Paso 1 (ETL 1): Ingesta de la oferta académica.

Lee el .xlsx, normaliza nombres (quita tildes y mapea alias),
descarta registros no homologables, exporta descartados a JSON
y reemplaza la colección en MongoDB.

Uso: python 01_ingest_excel.py
"""
from __future__ import annotations
from thefuzz import process, fuzz

import sys
from pathlib import Path
import json
import unicodedata

import pandas as pd
from pymongo.database import Database
from pymongo.errors import PyMongoError

from config import get_settings
from database import MongoConnection

REQUIRED_COLUMNS = (
    "id", "pais", "provincia", "ciudad", "universidad", "siglas", "nombre",
    "epn_homologa", "url", "descripcion", "perfil_ingreso", "perfil_egreso",
    "perfil_profesional",
)
NULL_LITERALS = {"", "nan", "none", "null", "n/a", "na", "no registra"}

# ---------------------------------------------------------
# 1. NORMALIZACIÓN Y CONFIGURACIÓN DE CARRERAS
# ---------------------------------------------------------
def limpiar_texto(texto: str) -> str:
    """Convierte a minúsculas, quita tildes y espacios extra."""
    if pd.isna(texto) or texto is None:
        return ""
    texto = str(texto).strip().lower()
    # Elimina tildes (Diacríticos)
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto

try:
    with open("data/carreras_interes.json", "r", encoding="utf-8") as f:
        lista_interes = json.load(f)
    # Guardamos los nombres originales para inyectarlos en MongoDB, indexados por su versión limpia
    MAPA_OFICIAL = {limpiar_texto(c): c for c in lista_interes}
except FileNotFoundError:
    print("[ERROR] No se encontró el archivo data/carreras_interes.json", file=sys.stderr)
    sys.exit(1)

# Diccionario de sinónimos/alias (Texto limpio -> Texto oficial limpio)
# ALIAS_CARRERAS = {
#     # Alias previos
#     "computacion": "ingenieria en ciencias de la computacion",
#     "electricidad": "ingenieria en electricidad",
#     "electronica y automatizacion": "ingenieria en electronica y automatizacion",
#     "telecomunicaciones": "ingenieria en telecomunicaciones",
#     "mecatronica": "ingenieria en mecatronica",
#     "mecanica": "ingenieria en mecanica",
#     "quimica": "ingenieria quimica",
#     "materiales": "ingenieria en materiales",
#     "petroleos": "ingenieria en petroleos",
#     "geologia": "ingenieria en geologia",
#     "matematica": "matematica",
#     "economia": "economia",
#     "administracion de empresas": "licenciatura en administracion de empresas",
#     "licenciatura administracion de empresas": "licenciatura en administracion de empresas",
#     "ingenieria ambiental": "ingenieria ambiental",
#     "ingenieria civil": "ingenieria civil",
#     "ingenieria industrial": "ingenieria de la produccion", 
#     "software": "ingenieria en software",
#     "sistemas de informacion": "ingenieria en sistemas de informacion",
#     "tecnologias de la informacion": "ingenieria en telecomunicacion de la informacion",
#     "ciencia de datos": "ciencias de datos e inteligencia artificial",
#     "ciencia de datos e inteligencia artificial": "ciencias de datos e inteligencia artificial",

    
#     # Nuevos alias basados en tus descartados
#     "tecnologia en desarrollo de software fullstack": "ingenieria en software",
#     "ingenieria en ciencia de datos con itinerario en analitica de datos": "ciencias de datos e inteligencia artificial",
#     "software (ingenieria)": "ingenieria en software",
#     "tecnologias de la informacion (ingenieria)": "ingenieria en telecomunicacion de la informacion",
#     "ingenieria automotriz": "ingenieria en mecanica",
#     "licenciatura en administracion de empresas": "licenciatura administracion de empresas"
# }

NULL_LITERALS = {"", "nan", "none", "null", "n/a", "na", "no registra", "no aplica"}

# Diccionario de sinónimos/alias (Texto limpio -> Texto oficial limpio)
ALIAS_CARRERAS = {
    "computacion": "ingenieria en ciencias de la computacion",
    "electricidad": "ingenieria en electricidad",
    "electronica y automatizacion": "ingenieria en electronica y automatizacion",
    "telecomunicaciones": "ingenieria en telecomunicaciones",
    "mecatronica": "ingenieria en mecatronica",
    "mecanica": "ingenieria en mecanica",
    "quimica": "ingenieria quimica",
    "materiales": "ingenieria en materiales",
    "petroleos": "ingenieria en petroleos",
    "geologia": "ingenieria en geologia",
    # CORREGIDO: Añadido el "en" para que coincida con tu JSON
    "administracion de empresas": "licenciatura en administracion de empresas",
    "ingenieria industrial": "ingenieria de la produccion", 
    "software": "ingenieria en software",
    "sistemas de informacion": "ingenieria en sistemas de informacion",
    # CORREGIDO: Ajustado al nuevo nombre de tu JSON
    "tecnologias de la informacion": "ingenieria en tecnologias de la informacion",
    "ciencia de datos": "ciencias de datos e inteligencia artificial",
    "ingenieria automotriz": "ingenieria en mecanica",
    # NUEVO: Alias para que atrape la homologa que viene en el Excel
    "ingenieria en ciencia de datos e inteligencia artificial": "ciencias de datos e inteligencia artificial",

    "agroindustria": "ingenieria agroindustria",

    "Ingenieria En Telecomunicacion De La Informacion": "ingenieria en tecnologias de la informacion",
}


# Configuración del umbral de similitud (85% es el estándar recomendado)
UMBRAL_FUZZY = 90

def buscar_mejor_coincidencia(texto_limpio: str) -> str | None:
    """Aplica las 3 capas de búsqueda: Exacta -> Alias -> Difusa (Estricta)."""
    if not texto_limpio or texto_limpio in NULL_LITERALS:
        return None
        
    # 1. Búsqueda Exacta
    if texto_limpio in MAPA_OFICIAL:
        return MAPA_OFICIAL[texto_limpio]
        
    # 2. Búsqueda por Alias Estructural
    if texto_limpio in ALIAS_CARRERAS and ALIAS_CARRERAS[texto_limpio] in MAPA_OFICIAL:
        return MAPA_OFICIAL[ALIAS_CARRERAS[texto_limpio]]
        
    # 3. Búsqueda Difusa ESTRICTA (fuzz.ratio)
    opciones = list(MAPA_OFICIAL.keys())
    mejor_match, puntuacion = process.extractOne(texto_limpio, opciones, scorer=fuzz.ratio)
    
    if puntuacion >= UMBRAL_FUZZY:
        return MAPA_OFICIAL[mejor_match]
        
    return None

def obtener_carrera_oficial(row) -> str | None:
    """Prioriza el nombre de la carrera. Si no coincide, recurre a epn_homologa."""
    nombre = limpiar_texto(row.get('nombre', ''))
    homologa = limpiar_texto(row.get('epn_homologa', ''))
    
    # Prioridad 1: Revisar el nombre de la carrera
    if nombre not in NULL_LITERALS:
        match_nombre = buscar_mejor_coincidencia(nombre)
        if match_nombre:
            return match_nombre
            
    # Si el nombre no coincidió (o estaba vacío), revisamos epn_homologa.
    # Si homologa está vacío, es 'null', 'no registra' o 'no aplica', se descarta.
    if homologa in NULL_LITERALS:
        return None
        
    # Prioridad 2: Revisar epn_homologa si es un valor válido
    match_homologa = buscar_mejor_coincidencia(homologa)
    if match_homologa:
        return match_homologa

    # Si ninguna de las dos columnas hizo match, se rechaza
    return None



# def obtener_carrera_oficial(row) -> str | None:
#     """Busca coincidencias normalizadas y devuelve el nombre oficial con mayúsculas correctas."""
#     nombre = limpiar_texto(row.get('nombre', ''))
#     homologa = limpiar_texto(row.get('epn_homologa', ''))
    
#     # 1. Intentar hacer match directo o por alias con el nombre de la carrera
#     # (Ignoramos el nombre si explícitamente dice 'no registra' en la homologa)
#     if homologa not in NULL_LITERALS:
#         if nombre in MAPA_OFICIAL:
#             return MAPA_OFICIAL[nombre]
#         if nombre in ALIAS_CARRERAS and ALIAS_CARRERAS[nombre] in MAPA_OFICIAL:
#             return MAPA_OFICIAL[ALIAS_CARRERAS[nombre]]
            
#         # 2. Intentar hacer match con el campo epn_homologa
#         if homologa in MAPA_OFICIAL:
#             return MAPA_OFICIAL[homologa]
#         if homologa in ALIAS_CARRERAS and ALIAS_CARRERAS[homologa] in MAPA_OFICIAL:
#             return MAPA_OFICIAL[ALIAS_CARRERAS[homologa]]
            
#     # Caso especial: Si homologa está vacía, pero el nombre sí es válido
#     elif homologa in NULL_LITERALS:
#         if nombre in MAPA_OFICIAL:
#             return MAPA_OFICIAL[nombre]
#         if nombre in ALIAS_CARRERAS and ALIAS_CARRERAS[nombre] in MAPA_OFICIAL:
#             return MAPA_OFICIAL[ALIAS_CARRERAS[nombre]]

#     return None

# ---------------------------------------------------------
# 2. LÓGICA DE INGESTA
# ---------------------------------------------------------

def _clean(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return None if text.lower() in NULL_LITERALS else text

def read_excel(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo Excel: {path}")

    df = pd.read_excel(path, dtype=str, engine="openpyxl")
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el Excel: {missing}")

    return df.apply(lambda col: col.map(_clean))

def filter_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {}

    # Generar el nuevo campo con la carrera estandarizada
    df['carrera_epn_oficial'] = df.apply(obtener_carrera_oficial, axis=1)
    
    # Filtrar: Si el nuevo campo es nulo, significa que no hizo match con nada (se descarta)
    mask_whitelist = df['carrera_epn_oficial'].notna()
    stats["descartados_lista_blanca"] = int((~mask_whitelist).sum())
    
    df_descartados = df[~mask_whitelist].copy()
    df_validos = df[mask_whitelist].copy()

    # Ya NO sobreescribimos epn_homologa. Ahora MongoDB tendrá los 3 campos:
    # 1. nombre (original)
    # 2. epn_homologa (original)
    # 3. carrera_epn_oficial (normalizado)

    mask_id = df_validos["id"].notna()
    stats["sin_id"] = int((~mask_id).sum())
    df_validos = df_validos[mask_id]

    before = len(df_validos)
    df_validos = df_validos.drop_duplicates(subset="id", keep="first")
    stats["id_duplicado"] = before - len(df_validos)

    return df_validos, df_descartados, stats


def load_collection(db: Database, name: str, documents: list[dict]) -> int:
    collection = db[name]
    collection.drop()
    if not documents:
        return 0

    result = collection.insert_many(documents, ordered=False)
    collection.create_index("id", unique=True)
    collection.create_index("ciudad")
    collection.create_index("epn_homologa")
    return len(result.inserted_ids)

def print_summary(read: int, stats: dict[str, int], inserted: int, stored: int, df: pd.DataFrame) -> None:
    line = "=" * 58
    print(line)
    print(" RESUMEN DE INGESTA - oferta_academica")
    print(line)
    print(f" Registros leídos del Excel ........ {read}")
    print(f" Descartados por lista blanca ...... {stats.get('descartados_lista_blanca', 0)}")
    print(f" Descartados sin id ................ {stats['sin_id']}")
    print(f" Descartados por id duplicado ...... {stats['id_duplicado']}")
    print(f" Registros insertados .............. {inserted}")
    print(f" Documentos en la colección ........ {stored}")
    if len(df):
        print(f" Universidades distintas ........... {df['universidad'].nunique()}")
        print(f" Ciudades distintas ................ {df['ciudad'].nunique()}")
    print(line)
    print(" ESTADO:", "OK" if inserted == stored == len(df) else "INCONSISTENTE - revisar")
    print(line)

def main() -> int:
    settings = get_settings()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    try:
        raw = read_excel(settings.excel_path)
        filtered, discarded, stats = filter_records(raw)
        
        # Exportar descartados
        if not discarded.empty:
            columnas_exportar = [col for col in ["id", "pais", "ciudad", "universidad", "nombre", "epn_homologa"] if col in discarded.columns]
            discarded[columnas_exportar].to_json("outputs/descartados_ingesta.json", orient="records", force_ascii=False, indent=4)

        if len(filtered) > 0:
            documents = filtered.to_dict("records")
            db = MongoConnection.get_db()
            inserted = load_collection(db, settings.coll_academic, documents)
            stored = db[settings.coll_academic].count_documents({})
        else:
            inserted = 0
            stored = 0

        print_summary(len(raw), stats, inserted, stored, filtered)
        if not discarded.empty:
            print(f" [INFO] Detalle de descartados guardado en: outputs/descartados_ingesta.json")
            print("=" * 58)
            
        return 0 if inserted == stored == len(filtered) else 1
        
    except (FileNotFoundError, ValueError, EnvironmentError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except PyMongoError as exc:
        print(f"[ERROR MongoDB] {exc}", file=sys.stderr)
        return 3
    finally:
        MongoConnection.close()

if __name__ == "__main__":
    sys.exit(main())
