"""
Финализация feature-based детектора v1 (ТОЛЬКО длинные тексты, >=200 симв).

Честная схема train/val/test:
  - train.jsonl  -> обучаем калиброванную модель
  - val.jsonl    -> ПОДБИРАЕМ порог под целевой FPR (таблица порог->FPR->recall)
  - test.jsonl   -> финальный замер РОВНО ОДИН РАЗ при выбранном пороге

Порог выбираем как минимальный T, при котором FPR(люди) на VAL <= target_fpr
(минимальный T -> максимальный recall среди удовлетворяющих ограничению на FPR).

Запуск:
    python finalize_v1.py [--target-fpr 0.03] [--model-out model_v1_long.joblib]
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from aidetector.dataset import build_feature_matrix, load_jsonl
from aidetector.features import FEATURE_NAMES
from aidetector.paths import model_path


def fpr_recall_at(y, proba, thr):
    """FPR (люди ошибочно как ИИ) и recall по ИИ при пороге thr."""
    pred = (proba >= thr).astype(int)
    human = y == 0
    ai = y == 1
    fpr = float(pred[human].mean()) if human.any() else 0.0          # FP/нег
    recall = float(pred[ai].mean()) if ai.any() else 0.0              # TP/поз
    return fpr, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.03)
    ap.add_argument("--model-out", default=model_path("model_v1_long.joblib"))
    args = ap.parse_args()

    print("Извлечение признаков (train/val/test)...")
    Xtr, ytr = build_feature_matrix(load_jsonl("data/train.jsonl"))
    Xval, yval = build_feature_matrix(load_jsonl("data/val.jsonl"))
    Xte, yte = build_feature_matrix(load_jsonl("data/test.jsonl"))
    print(f"train={len(ytr)}  val={len(yval)}  test={len(yte)}")

    base = GradientBoostingClassifier(random_state=42)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr, ytr)

    pval = clf.predict_proba(Xval)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]

    # --- таблица порогов на VALIDATION ---
    print("\n=== ПОДБОР ПОРОГА на validation ===")
    print(f"{'порог':>6} | {'FPR люди':>9} | {'recall ИИ':>10}")
    print("-" * 32)
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        fpr, rec = fpr_recall_at(yval, pval, thr)
        print(f"{thr:>6.2f} | {fpr:>9.3f} | {rec:>10.3f}")

    # авто-выбор: минимальный T с FPR_val <= target (макс. recall при FPR<=target)
    grid = np.round(np.arange(0.05, 0.991, 0.005), 3)
    chosen = None
    for thr in grid:                      # по возрастанию: FPR падает с ростом thr
        fpr, _ = fpr_recall_at(yval, pval, thr)
        if fpr <= args.target_fpr:
            chosen = float(thr)
            break
    if chosen is None:
        chosen = float(grid[-1])
    fpr_v, rec_v = fpr_recall_at(yval, pval, chosen)
    print(f"\nВыбран порог T={chosen:.3f}  (на VAL: FPR={fpr_v:.3f} <= {args.target_fpr}, "
          f"recall ИИ={rec_v:.3f})")

    # --- ФИНАЛЬНЫЙ замер на TEST (один раз) при выбранном пороге ---
    print("\n=== ФИНАЛ на held-out TEST (один раз), пороги длинные >=200 симв ===")
    auc = roc_auc_score(yte, pte)
    fpr_t, rec_t = fpr_recall_at(yte, pte, chosen)
    acc = float(((pte >= chosen).astype(int) == yte).mean())
    print(f"ROC-AUC:            {auc:.3f}")
    print(f"порог T:            {chosen:.3f}")
    print(f"FPR (люди):         {fpr_t:.3f}")
    print(f"recall по ИИ:       {rec_t:.3f}")
    print(f"accuracy@T:         {acc:.3f}")

    print("\n=== Калибровка на TEST (predicted ≈ observed) ===")
    frac_pos, mean_pred = calibration_curve(yte, pte, n_bins=5, strategy="quantile")
    print(f"  {'предсказано':>12} | {'факт ИИ':>10}")
    for mp, fpos in zip(mean_pred, frac_pos):
        print(f"  {mp:>12.2f} | {fpos:>10.2f}")

    # важность фич
    imps = []
    for cc in getattr(clf, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None and hasattr(est, "feature_importances_"):
            imps.append(est.feature_importances_)
    importances = np.mean(imps, axis=0) if imps else np.zeros(len(FEATURE_NAMES))
    print("\n=== Важность фич ===")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda t: t[1], reverse=True):
        print(f"  {name:20s} {imp:.3f}")

    joblib.dump(clf, args.model_out)
    joblib.dump(clf, model_path("model.joblib"))   # рабочая модель для app.py
    with open(model_path("threshold_v1.json"), "w") as f:
        json.dump({"threshold": chosen, "target_fpr": args.target_fpr,
                   "test_fpr": fpr_t, "test_recall_ai": rec_t, "roc_auc": auc}, f, indent=2)
    print(f"\nСохранено: {args.model_out} (+ model.joblib), порог -> threshold_v1.json")


if __name__ == "__main__":
    main()
