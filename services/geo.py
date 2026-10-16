"""Normalización geográfica para cruzar `ciudad` de oferta_academica con gold_jobs."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Alias explícitos: ciudad normalizada de oferta_academica -> ciudades de gold_jobs.
# SUPUESTO A CONFIRMAR: "Manabi" es una provincia, no una ciudad. Se agrupa en las
# dos ciudades de Manabí presentes en gold_jobs. Elimina esta entrada para desactivarlo.
CITY_ALIASES = {
    "manabi": ("Manta", "Portoviejo"),
}


def normalize_key(text: str) -> str:
    """Minúsculas, sin tildes y con espacios colapsados: 'Durán ' -> 'duran'."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", no_accents).strip().casefold()


def build_city_index(gold_cities: Iterable) -> dict:
    """Índice {clave_normalizada: [grafías exactas en gold_jobs]} (ignora nulos y no-strings)."""
    index: dict = {}
    for city in gold_cities:
        if isinstance(city, str) and city.strip():
            spellings = index.setdefault(normalize_key(city), [])
            if city not in spellings:
                spellings.append(city)
    return index


def resolve_city(city: str, index: dict) -> list:
    """Devuelve las grafías exactas de gold_jobs equivalentes a `city` (lista vacía si no hay match)."""
    key = normalize_key(city)
    targets = CITY_ALIASES.get(key)
    if targets is None:
        return list(index.get(key, []))

    resolved: list = []
    for target in targets:
        for spelling in index.get(normalize_key(target), []):
            if spelling not in resolved:
                resolved.append(spelling)
    return resolved