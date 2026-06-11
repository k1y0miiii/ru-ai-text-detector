#!/usr/bin/env python3
"""
transformer_score.py — инференс трансформера: текст -> КАЛИБРОВАННАЯ p(ИИ).

ВЕРСИИ. Поддерживаем несколько обученных трансформеров; ДЕФОЛТ — v4 (боевая).
  v4 — v3-база ruBert-base (178M), ПЕРЕОБУЧЕННАЯ на расширенном train (+ИИ Llama 3.1 /
       Mistral 7B). Бьёт v3 по покрытию генераторов (holdout llama 0.45->0.94,
       mistral 0.51->0.92) и слегка по нетронутому gpt4 (0.74->0.77); test-FPR 0.023.
  v3 — дообученный ai-forever/ruBert-base (178M); оставлен как ОТКАТ (консервативнее,
       test-FPR 0.007). Был боевым до v4.
  v2 — дообученный cointegrated/rubert-tiny2 (29M), оставлен как ОТКАТ (быстрее).
Каждая версия: <ver>_model/ + <ver>_calibrator.joblib + threshold_<ver>.json,
кэш p по md5 в data/<ver>_score_cache.json. Считаем на CPU (как прод без GPU).

LENGTH-AWARE ПОРОГ. threshold_for(n_words): для коротких фрагментов (<cutoff слов)
отдельный СТРОЖЕ порог (threshold_short) — на коротком хвост FPR нестабилен, и единый
порог даёт ~3.8% вместо <=3%. Для длинных — базовый порог. Так FPR<=3% на ВСЕХ длинах.

Совместимость: module-level proba()/threshold() работают как раньше, но теперь это
ДЕФОЛТНАЯ версия (v3). Для конкретной версии — get("v2") / get("v3").
"""

import hashlib
import json
import os
import re

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .paths import data_path, model_path

MAXLEN = 256
DEFAULT_VERSION = "v4"

# реестр версий -> артефакты
# v4 — БОЕВАЯ (дефолт): v3-база ruBert-base, ПЕРЕОБУЧЕННАЯ на расширенном train с ИИ
#      Llama 3.1 + Mistral 7B. Приёмка 6/6: holdout llama 0.45->0.94, mistral 0.51->0.92,
#      gpt4 (нетронутый) 0.74->0.77, saiga 0.90->0.82(>=0.78), test-FPR 0.023(<=3%),
#      test_diploma 0/21. Компромисс: чуть агрессивнее на реальном ВКР (вердикт 1->3,
#      подсветка 6->24 из 374 абз. прозы) — флагует ШАБЛОННУЮ прозу, не оригинальную.
# v3 — оставлена как ОТКАТ (rollback): была боевой; консервативнее (test-FPR 0.007).
#      Артефакты v3 НЕ тронуты. Откат к v3 = поменять DEFAULT_VERSION обратно на "v3".
# v2 — rubert-tiny2 (29M), лёгкий быстрый откат.
REGISTRY = {
    "v2": dict(model=model_path("v2_model"), calib=model_path("v2_calibrator.joblib"),
               thr=model_path("threshold_v2.json")),
    "v3": dict(model=model_path("v3_model"), calib=model_path("v3_calibrator.joblib"),
               thr=model_path("threshold_v3.json")),
    "v4": dict(model=model_path("v4_model"), calib=model_path("v4_calibrator.joblib"),
               thr=model_path("threshold_v4.json")),
}
# алиас: «v3_legacy» = те же артефакты v3 (по ТЗ — v3 переименована в legacy; ключ v3
# сохранён для обратной совместимости ?model=v3 / --model v3)
REGISTRY["v3_legacy"] = REGISTRY["v3"]

# «слово» = токен с буквой (как в document_scan) — для length-aware порога
_WORD_RE = re.compile(r"\w*[^\W\d_]\w*", re.UNICODE)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


class Scorer:
    """Один трансформер-скорер версии ver. Ленивая загрузка, кэш p на диске."""

    def __init__(self, version: str):
        if version not in REGISTRY:
            raise ValueError(f"неизвестная версия {version}; есть {list(REGISTRY)}")
        self.version = version
        self._cfg = REGISTRY[version]
        self._m = self._t = self._iso = None
        self._thr = self._thr_short = self._cutoff = None
        self._cache = None
        self._cache_path = data_path(f"{version}_score_cache.json")

    def _ensure(self):
        if self._m is None:
            self._t = AutoTokenizer.from_pretrained(self._cfg["model"])
            self._m = (AutoModelForSequenceClassification
                       .from_pretrained(self._cfg["model"]).to("cpu").eval())
            self._iso = joblib.load(self._cfg["calib"])
            j = json.load(open(self._cfg["thr"], encoding="utf-8"))
            self._thr = float(j["threshold"])
            self._thr_short = float(j.get("threshold_short", j["threshold"]))
            self._cutoff = int(j.get("short_word_cutoff", 0))

    # --- пороги ---
    def threshold(self) -> float:
        """Базовый вердиктный порог (для текста нормальной длины)."""
        self._ensure()
        return self._thr

    def threshold_for(self, n_words: int) -> float:
        """Length-aware вердиктный порог: строже на коротком (<cutoff слов)."""
        self._ensure()
        if self._cutoff and n_words < self._cutoff:
            return self._thr_short
        return self._thr

    # --- кэш ---
    def _cload(self):
        if self._cache is None:
            self._cache = (json.load(open(self._cache_path, encoding="utf-8"))
                           if os.path.exists(self._cache_path) else {})
        return self._cache

    @staticmethod
    def _md5(t: str) -> str:
        return hashlib.md5(t.encode("utf-8")).hexdigest()

    @torch.no_grad()
    def proba(self, texts, bs: int = 32) -> np.ndarray:
        """Калиброванная p(ИИ) для списка текстов (кэш по md5)."""
        self._ensure()
        c = self._cload()
        need = list(dict.fromkeys(t for t in texts if self._md5(t) not in c))
        for i in range(0, len(need), bs):
            chunk = need[i:i + bs]
            enc = self._t(chunk, truncation=True, max_length=MAXLEN, padding=True,
                          return_tensors="pt")
            raw = torch.softmax(self._m(**enc).logits, dim=1)[:, 1].numpy()
            for t, p in zip(chunk, self._iso.predict(raw)):
                c[self._md5(t)] = float(p)
        if need:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            json.dump(c, open(self._cache_path, "w"))
        return np.array([c[self._md5(t)] for t in texts])


_scorers: dict = {}


def get(version: str = DEFAULT_VERSION) -> Scorer:
    """Кэшированный скорер версии (по умолчанию v3 — боевая)."""
    if version not in _scorers:
        _scorers[version] = Scorer(version)
    return _scorers[version]


# --- backward-compat: module-level функции = ДЕФОЛТНАЯ версия (v3) ---
def proba(texts, bs: int = 32) -> np.ndarray:
    return get().proba(texts, bs=bs)


def threshold() -> float:
    return get().threshold()


def threshold_for(n_words: int) -> float:
    return get().threshold_for(n_words)
