#!/usr/bin/env python3
"""
calibrate_v4.py — деплой-артефакты v4: изотоника на VAL + length-aware пороги
(база + строже для коротких <cutoff слов), как у v3. Сохраняет v4_calibrator.joblib
и threshold_v4.json. НЕ трогает прод (transformer_score/app/v3) — только после приёмки.

Процедура (идентична духу v3, чтобы сравнение было яблоко-к-яблоку):
  - изотоника фитится на сыром softmax VAL;
  - база: наименьший КАЛИБРОВАННЫЙ порог с full-length VAL-human FPR <= 2.5%;
  - short (cutoff=40 слов): VAL-human урезаем до 30 слов (репрезентативно для <40),
    берём наименьший калиброванный порог с FPR <= 2.5% на этих коротких;
  - highlight_threshold=0.6 (как v3 — мягкая подсветка карты).
Цель VAL FPR <= 2.5% с запасом под боевые 3%.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json

import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aidetector.document_scan import _WORD_RE
from aidetector.paths import model_path

MAXLEN = 256
CUTOFF = 40          # как у v3
SHORT_REPR = 30      # репрезентативная короткая длина для калибровки short-порога


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def trunc(t, n):
    sp = [(m.start(), m.end()) for m in _WORD_RE.finditer(t)]
    return t.strip() if len(sp) <= n else t[:sp[n - 1][1]].strip()


def smallest_thr(cal_human, grid, target=0.025):
    return next((float(t) for t in grid if float((cal_human >= t).mean()) <= target),
                float(grid[-1]))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path("v4_model"))
    m = AutoModelForSequenceClassification.from_pretrained(model_path("v4_model")).to(dev).eval()

    @torch.no_grad()
    def raw(texts, bs=64):
        ps = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to(dev)
            ps.append(torch.softmax(m(**enc).logits, 1)[:, 1].float().cpu().numpy())
        return np.concatenate(ps)

    val = load("data/v2_val.jsonl")
    vt = [r["text"] for r in val]
    vy = np.array([r["label"] for r in val])
    vr = raw(vt)

    iso = IsotonicRegression(out_of_bounds="clip").fit(vr, vy)
    vc = iso.predict(vr)

    grid = np.round(np.arange(0.05, 0.9991, 0.001), 3)
    hum_full = vc[vy == 0]
    thr = smallest_thr(hum_full, grid)
    val_fpr = float((hum_full >= thr).mean())
    val_rec = float((vc[vy == 1] >= thr).mean())

    # short-порог: VAL-human урезаем до SHORT_REPR слов, калибруем p, наим. порог FPR<=2.5%
    val_h_texts = [r["text"] for r in val if r["label"] == 0]
    short_raw = raw([trunc(t, SHORT_REPR) for t in val_h_texts])
    short_cal = iso.predict(short_raw)
    thr_short = smallest_thr(short_cal, grid)
    short_fpr = float((short_cal >= thr_short).mean())

    joblib.dump(iso, model_path("v4_calibrator.joblib"))
    out = {
        "threshold": thr,
        "threshold_short": thr_short,
        "short_word_cutoff": CUTOFF,
        "base": "ai-forever/ruBert-base (v4 = v3 + llama3.1/mistral в train)",
        "calibration": "isotonic on VAL",
        "selection": "наименьший калиброванный порог с VAL-human FPR<=2.5%",
        "val_fpr": val_fpr,
        "val_recall": val_rec,
        "val_short_fpr_at_30w": short_fpr,
        "highlight_threshold": 0.6,
        "note": ("Вердиктный порог для >=40 слов; для <40 слов threshold_short строже "
                 "(держит FPR на коротком хвосте). highlight_threshold=0.6 — мягкая "
                 "подсветка карты. Деривация идентична v3 для яблоко-к-яблоку сравнения."),
    }
    json.dump(out, open(model_path("threshold_v4.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[calibrate v4] база={thr:.4f} (VAL FPR={val_fpr:.4f}, recall={val_rec:.4f}) | "
          f"short={thr_short:.4f} (VAL-30w FPR={short_fpr:.4f}, cutoff={CUTOFF})")
    print("[saved] v4_calibrator.joblib, threshold_v4.json")


if __name__ == "__main__":
    main()
