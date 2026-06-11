#!/usr/bin/env python3
"""
train_v1_1.py — переобучение feature-based детектора на расширенных данных (v1.1).

ТОЛЬКО данные новые. Архитектура/признаки/схема порога — РОВНО как в finalize_v1.py:
GradientBoosting в CalibratedClassifierCV(isotonic, cv=5); порог = минимальный T с
FPR(val)<=target. Выходы ОТДЕЛЬНЫЕ — model_v1_1.joblib / threshold_v1_1.json
(v1: model.joblib / model_v1_long.joblib / threshold_v1.json НЕ трогаем).
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from aidetector.dataset import load_jsonl
from aidetector.featcache import vectors
from aidetector.paths import model_path


def mat(rows):
    X = np.array(vectors([r["text"] for r in rows]), dtype=np.float32)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    return X, y


def fpr_recall_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    fpr = float(pred[y == 0].mean()) if (y == 0).any() else 0.0
    rec = float(pred[y == 1].mean()) if (y == 1).any() else 0.0
    return fpr, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.03)
    args = ap.parse_args()

    print("Фичи (train_v1_1/val/test, с кэшем)...")
    Xtr, ytr = mat(load_jsonl("data/train_v1_1.jsonl"))
    Xval, yval = mat(load_jsonl("data/val.jsonl"))
    Xte, yte = mat(load_jsonl("data/test.jsonl"))
    print(f"train={len(ytr)}  val={len(yval)}  test={len(yte)}")

    base = GradientBoostingClassifier(random_state=42)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr, ytr)

    pval = clf.predict_proba(Xval)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]

    # порог: минимальный T с FPR(val) <= target (как в finalize_v1)
    grid = np.round(np.arange(0.05, 0.991, 0.005), 3)
    chosen = next((float(t) for t in grid if fpr_recall_at(yval, pval, t)[0] <= args.target_fpr),
                  float(grid[-1]))
    fpr_v, rec_v = fpr_recall_at(yval, pval, chosen)
    print(f"Порог T={chosen:.3f} (VAL: FPR={fpr_v:.3f}<= {args.target_fpr}, recall={rec_v:.3f})")

    auc = roc_auc_score(yte, pte)
    fpr_t, rec_t = fpr_recall_at(yte, pte, chosen)
    print("\n=== v1.1 на held-out TEST (свои генераторы) ===")
    print(f"ROC-AUC {auc:.3f} | порог {chosen:.3f} | FPR {fpr_t:.3f} | recall ИИ {rec_t:.3f}")

    # генерализация: recall на hold-out GPT-4 / saiga (НЕ в train)
    print("\n=== Генерализация (recall на невиданном, порог v1.1) ===")
    for name, path in [("GPT-4", "data/holdout_gpt4.jsonl"), ("saiga", "data/holdout_saiga.jsonl")]:
        Xh, yh = mat(load_jsonl(path))
        ph = clf.predict_proba(Xh)[:, 1]
        print(f"  {name:6s} n={len(yh)}: recall={(ph>=chosen).mean():.3f}  (median p={np.median(ph):.3f})")

    joblib.dump(clf, model_path("model_v1_1.joblib"))
    with open(model_path("threshold_v1_1.json"), "w") as f:
        json.dump({"threshold": chosen, "target_fpr": args.target_fpr,
                   "test_fpr": fpr_t, "test_recall_ai": rec_t, "roc_auc": auc}, f, indent=2)
    print("\nСохранено: model_v1_1.joblib + threshold_v1_1.json (v1 не тронут)")


if __name__ == "__main__":
    main()
