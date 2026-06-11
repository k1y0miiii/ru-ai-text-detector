#!/usr/bin/env python3
"""
compare_v1_v1_1.py — честное сравнение v1 vs v1.1 на ОДНИХ И ТЕХ ЖЕ данных/метриках.
Оба детектора — одни features/архитектура, отличие только в обучающих данных и пороге.
Ничего не переобучаем, модели уже сохранены. Фичи — через общий кэш.
"""

import json

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from aidetector.featcache import vectors
from aidetector.perturb import perturb
from aidetector.paths import model_path


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def proba(clf, texts):
    X = np.array(vectors(texts), dtype=np.float32)
    return clf.predict_proba(X)[:, 1]


def recall(p, thr):
    return float((np.asarray(p) >= thr).mean()) if len(p) else 0.0


V1 = joblib.load(model_path("model.joblib"))
T1 = 0.86
V11 = joblib.load(model_path("model_v1_1.joblib"))
T11 = json.load(open(model_path("threshold_v1_1.json")))["threshold"]
print(f"v1 порог={T1}  v1.1 порог={T11}\n")

# ---- held-out TEST (свои генераторы) ----
te = load("data/test.jsonl")
txt = [r["text"] for r in te]
y = np.array([r["label"] for r in te])
p1, p11 = proba(V1, txt), proba(V11, txt)


def block(name, y, p1, p11):
    a1, a11 = roc_auc_score(y, p1), roc_auc_score(y, p11)
    f1 = recall(p1[y == 0], T1); f11 = recall(p11[y == 0], T11)
    r1 = recall(p1[y == 1], T1); r11 = recall(p11[y == 1], T11)
    print(f"  {name}")
    print(f"    ROC-AUC:     v1 {a1:.3f}   v1.1 {a11:.3f}")
    print(f"    FPR (люди):  v1 {f1:.3f}   v1.1 {f11:.3f}")
    print(f"    recall ИИ:   v1 {r1:.3f}   v1.1 {r11:.3f}")


print("=== TEST (свои генераторы chatgpt/qwen3b/coat, n={}) ===".format(len(te)))
block("общий", y, p1, p11)

# ---- генерализация: hold-out GPT-4 / saiga (НЕ в train ни v1, ни v1.1) ----
print("\n=== ГЕНЕРАЛИЗАЦИЯ: recall на невиданном ИИ (только AI, p>=порог) ===")
for nm, path in [("GPT-4 (нет у v1; ДОБАВЛЕН в v1.1)", "data/holdout_gpt4.jsonl"),
                 ("saiga gpt-3.5 (нет у v1; ДОБАВЛЕН в v1.1)", "data/holdout_saiga.jsonl")]:
    rows = load(path)
    t = [r["text"] for r in rows]
    q1, q11 = proba(V1, t), proba(V11, t)
    print(f"  {nm:42s} n={len(rows)}: v1 {recall(q1,T1):.3f} -> v1.1 {recall(q11,T11):.3f}")

# ---- робастность к правкам (тот же батч, perturb вслепую) ----
print("\n=== РОБАСТНОСТЬ к правкам (measure_robustness, тот же perturb) ===")
rob = load("data/robustness_raw.jsonl")
raw = [r["text"] for r in rob]
edt = [perturb(r["text"]) for r in rob]
buckets = np.array([r["bucket"] for r in rob])
rb1, ra1 = proba(V1, raw), proba(V1, edt)
rb11, ra11 = proba(V11, raw), proba(V11, edt)


def rob_line(tag, mask):
    b1 = recall(rb1[mask], T1); a1 = recall(ra1[mask], T1)
    b11 = recall(rb11[mask], T11); a11 = recall(ra11[mask], T11)
    print(f"  {tag:28s} v1: {b1:.3f}->{a1:.3f} ({(b1-a1)*100:+.1f}пп)   "
          f"v1.1: {b11:.3f}->{a11:.3f} ({(b11-a11)*100:+.1f}пп)")


rob_line(f"весь батч (n={len(rob)})", np.ones(len(rob), bool))
rob_line("A: свои генераторы", buckets == "A")
rob_line("B: чужой ИИ (saiga/gpt4)", buckets == "B")

print("\n[готово] сырой/правленый recall выше = «сырой -> после правок»")
