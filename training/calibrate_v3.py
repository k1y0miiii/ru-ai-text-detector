#!/usr/bin/env python3
"""
calibrate_v3.py — деплой-артефакты v3: изотоника на VAL + единый порог под FPR<=2.5%
на VAL (как у v2). Сохраняет v3_calibrator.joblib + threshold_v3.json. НЕ трогает
transformer_score.py / app.py / v2 — v3 идёт в прод только после приёмки.
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
from aidetector.paths import model_path

MAXLEN = 256


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path("v3_model"))
    m = AutoModelForSequenceClassification.from_pretrained(model_path("v3_model")).to(dev).eval()

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

    # единый порог: наименьший с VAL-FPR <= 2.5% (на КАЛИБРОВАННОЙ p, как v2)
    grid = np.round(np.arange(0.05, 0.9991, 0.001), 3)
    hum = vc[vy == 0]
    thr = next((float(t) for t in grid if float((hum >= t).mean()) <= 0.025), float(grid[-1]))
    val_fpr = float((hum >= thr).mean())
    val_rec = float((vc[vy == 1] >= thr).mean())

    joblib.dump(iso, model_path("v3_calibrator.joblib"))
    json.dump({"threshold": thr, "base": "ruBert-base (v3)", "val_fpr": val_fpr,
               "val_recall": val_rec, "selection": "smallest VAL thr with FPR<=0.025"},
              open(model_path("threshold_v3.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[calibrate v3] порог={thr:.3f}  VAL FPR={val_fpr:.3f}  VAL recall={val_rec:.3f}")
    print("[saved] v3_calibrator.joblib, threshold_v3.json")


if __name__ == "__main__":
    main()
