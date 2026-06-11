#!/usr/bin/env python3
"""
docx_v2.py — три наших docx (mixed/pure_ai/ai_edited) на трансформере v2 vs v1.
Содержательные абзацы (после того же отсева classify_sections) скорим НАПРЯМУЮ:
трансформер читает абзац любой длины (окно v1 ему не нужно). Главное — слепнет ли
v2 на отредактированном ИИ (ai_edited), как v1 (фундаментальный предел или нет).
"""

import joblib
import numpy as np

from aidetector import transformer_score as TF
from aidetector.document_scan import classify_sections, split_paragraphs
from aidetector.extract_text import extract_text
from aidetector.featcache import vectors
from aidetector.paths import model_path

V1 = joblib.load(model_path("model.joblib"))
T1, T2 = 0.86, TF.threshold()


def v1p(texts):
    return V1.predict_proba(np.array(vectors(texts), dtype=np.float32))[:, 1] if texts else np.array([])


for doc in ["test_mixed", "test_pure_ai", "test_ai_edited"]:
    text, _ = extract_text(f"samples/{doc}.docx")
    content = classify_sections(split_paragraphs(text)).content
    cps = [p.text for p in content if len(p.text) >= 120]   # содержательные абзацы
    p2 = TF.proba(cps)
    longs = [t for t in cps if len(t) >= 200]
    p1 = v1p(longs)
    print(f"\n=== {doc}.docx ===  содержательных абзацев: {len(cps)}")
    print(f"  v2: подозрительно {int((p2>=T2).sum())}/{len(cps)} ({(p2>=T2).mean():.0%}), "
          f"median p={np.median(p2):.3f}")
    if len(longs):
        print(f"  v1 (только >=200, n={len(longs)}): подозрительно {int((p1>=T1).sum())}/{len(longs)} "
              f"({(p1>=T1).mean():.0%}), median p={np.median(p1):.3f}")
