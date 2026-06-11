#!/usr/bin/env python3
"""
featcache.py — кэш фич на диске (md5(text) -> вектор), чтобы train и eval скрипты
v1.1 не пересчитывали features.extract по многу раз для одних и тех же текстов.

Features/архитектуру НЕ меняем — это просто кэш поверх тех же features.extract/to_vector.
"""

import hashlib
import json
import os

from .features import extract, to_vector

CACHE = "data/feat_cache.json"
_cache = None


def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load():
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    return _cache


def _save():
    json.dump(_cache, open(CACHE, "w", encoding="utf-8"))


def vectors(texts, save_every: int = 100):
    """Список фич-векторов для texts (порядок сохранён), с кэшированием на диск."""
    c = _load()
    new = 0
    for t in texts:
        k = _key(t)
        if k not in c:
            c[k] = [float(x) for x in to_vector(extract(t))]
            new += 1
            if new % save_every == 0:
                _save()
                print(f"  featurized {new} new (cache {len(c)})")
    if new:
        _save()
    return [c[_key(t)] for t in texts]
