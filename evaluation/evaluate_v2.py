#!/usr/bin/env python3
"""
evaluate_v2.py — честное сравнение v1 (feature-based) vs v2 (трансформер) vs
АНСАМБЛЬ на ОДНИХ И ТЕХ ЖЕ данных. Главный критерий — recall на НЕВИДАННЫХ
генераторах (hold-out GPT-4 / saiga), которых нет в train ни у кого.

Ничего не переобучаем. v1: features.extract+model.joblib (порог 0.86). v2:
transformer_score (калиброванная p, порог из threshold_v2.json). Ансамбль:
короткие (<200) -> v2; длинные (>=200) -> среднее калиброванных p(v1,v2).
"""

import json

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from aidetector import transformer_score as TF
from aidetector.featcache import vectors
from aidetector.perturb import perturb
from aidetector.paths import model_path

# Этот скрипт сравнивает именно v2 (трансформер-предшественник). Дефолт TF — теперь v3,
# поэтому ЯВНО берём v2-скорер, чтобы цифры остались про v2.
TF2 = TF.get("v2")
V1 = joblib.load(model_path("model.joblib"))
T1 = 0.86
T2 = TF2.threshold()


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def v1p(texts):
    return V1.predict_proba(np.array(vectors(texts), dtype=np.float32))[:, 1]


def ens(texts, p1, p2):
    lens = np.array([len(t) for t in texts])
    return np.where(lens < 200, p2, 0.5 * (p1 + p2))


def rec(p, thr):
    return float((np.asarray(p) >= thr).mean()) if len(p) else 0.0


# --- порог ансамбля под FPR<=3% на v2_val ---
va = load("data/v2_val.jsonl")
yva = np.array([r["label"] for r in va])
vt = [r["text"] for r in va]
ep_va = ens(vt, v1p(vt), TF2.proba(vt))
grid = np.round(np.arange(0.05, 0.991, 0.005), 3)
TE = next((float(t) for t in grid if rec(ep_va[yva == 0], t) <= 0.03), float(grid[-1]))
print(f"пороги: v1={T1}  v2={T2:.3f}  ансамбль={TE:.3f}\n")


def score_set(path, label_field="label"):
    rows = load(path)
    t = [r["text"] for r in rows]
    y = np.array([r[label_field] for r in rows])
    p1, p2 = v1p(t), TF2.proba(t)
    pe = ens(t, p1, p2)
    return y, p1, p2, pe


# ====================== КРИТЕРИЙ №1: невиданные генераторы ======================
print("=" * 70)
print("  КРИТЕРИЙ №1 — recall на НЕВИДАННОМ ИИ (ни у кого в train)")
print("=" * 70)
print(f"  {'набор':22s} {'n':>4}  {'v1':>6} {'v2':>6} {'ens':>6}   (было v1=0.13/0.22)")
for name, path in [("GPT-4 (hold-out)", "data/holdout_gpt4.jsonl"),
                   ("saiga gpt-3.5 (hold-out)", "data/holdout_saiga.jsonl")]:
    y, p1, p2, pe = score_set(path)
    print(f"  {name:22s} {len(y):>4}  {rec(p1,T1):>6.3f} {rec(p2,T2):>6.3f} {rec(pe,TE):>6.3f}")

# ====================== TEST (свои генераторы, >=200) ======================
print("\n=== TEST свои генераторы (>=200) — не просел ли на своих ===")
y, p1, p2, pe = score_set("data/test.jsonl")
for tag, p, thr in [("v1", p1, T1), ("v2", p2, T2), ("ens", pe, TE)]:
    print(f"  {tag:4s}: ROC-AUC {roc_auc_score(y,p):.3f}  FPR {rec(p[y==0],thr):.3f}  "
          f"recall ИИ {rec(p[y==1],thr):.3f}")

# ====================== КОРОТКИЕ (<200) — слепой угол v1 ======================
print("\n=== КОРОТКИЕ тексты (<200) — где v1 был слеп (recall 0.39/FPR 0.24) ===")
y, p1, p2, pe = score_set("data/short_test.jsonl")
for tag, p, thr in [("v1", p1, T1), ("v2", p2, T2), ("ens", pe, TE)]:
    print(f"  {tag:4s}: ROC-AUC {roc_auc_score(y,p):.3f}  FPR {rec(p[y==0],thr):.3f}  "
          f"recall ИИ {rec(p[y==1],thr):.3f}")

# ====================== РОБАСТНОСТЬ к правкам ======================
print("\n=== РОБАСТНОСТЬ к лёгкому редактированию (perturb, тот же батч) ===")
rob = load("data/robustness_raw.jsonl")
raw = [r["text"] for r in rob]
edt = [perturb(r["text"]) for r in rob]
r1b, r1a = v1p(raw), v1p(edt)
r2b, r2a = TF2.proba(raw), TF2.proba(edt)
reb = ens(raw, r1b, r2b); rea = ens(edt, r1a, r2a)
for tag, b, a, thr in [("v1", r1b, r1a, T1), ("v2", r2b, r2a, T2), ("ens", reb, rea, TE)]:
    print(f"  {tag:4s}: recall {rec(b,thr):.3f} -> {rec(a,thr):.3f}  "
          f"({(rec(b,thr)-rec(a,thr))*100:+.1f} п.п.)")

print("\n[готово]")
