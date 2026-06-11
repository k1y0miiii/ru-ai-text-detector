#!/usr/bin/env python3
"""
recalibrate_v2.py — ТОЛЬКО перекалибровка порога v2 (модель НЕ трогаем).

Проблема: v2 на test даёт FPR 0.045 при цели <=3% — нарушает главный инвариант
продукта (не обвинять людей). Старый порог 0.755 выбран на VAL под FPR<=3%, но
зазор VAL->test пробил потолок на test.

Решение: выбрать порог на VAL с ЗАПАСОМ (FPR<=2.5%), чтобы на test остаться <=3%.
Отбор порога — ТОЛЬКО на VAL. Замер на test — ОДИН раз. То же для ансамбля.

Ничего не переобучаем: v2_model/ + v2_calibrator.joblib неизменны, меняется лишь
порог в threshold_v2.json.
"""

import json

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from aidetector import transformer_score as TF
from aidetector.featcache import vectors
from aidetector.paths import model_path

TARGET_VAL_FPR = 0.025          # запас под продуктовый потолок 3% на test
OLD_THR = 0.755                 # текущий порог v2 (для замера цены ужесточения)

V1 = joblib.load(model_path("model.joblib"))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def v1p(texts):
    return V1.predict_proba(np.array(vectors(texts), dtype=np.float32))[:, 1]


def ens(texts, p1, p2):
    """Ансамбль: короткие (<200) -> только v2; длинные -> среднее p(v1,v2)."""
    lens = np.array([len(t) for t in texts])
    return np.where(lens < 200, p2, 0.5 * (p1 + p2))


def fpr(p, y, thr):
    h = p[y == 0]
    return float((h >= thr).mean()) if len(h) else 0.0


def rec(p, y, thr):
    a = p[y == 1]
    return float((a >= thr).mean()) if len(a) else 0.0


def pick_threshold(p, y, target):
    """Наименьший порог с VAL FPR <= target (максимизируем recall при потолке FPR)."""
    grid = np.round(np.arange(0.05, 0.991, 0.005), 3)
    return next((float(t) for t in grid if fpr(p, y, t) <= target), float(grid[-1]))


def val_table(name, p, y, target):
    nH = int((y == 0).sum())
    print(f"\n  === {name}: VAL (human={nH}, шаг FPR = {1/nH*100:.2f}%) ===")
    print(f"  {'thr':>5} {'FPR':>7} {'n_fp':>5} {'recall':>7}")
    chosen = pick_threshold(p, y, target)
    for thr in np.round(np.arange(0.60, 0.951, 0.01), 3):
        f = fpr(p, y, thr)
        r = rec(p, y, thr)
        nfp = int((p[y == 0] >= thr).sum())
        mark = "  <-- ВЫБРАН (<=%.1f%%)" % (target * 100) if abs(thr - chosen) < 1e-9 else ""
        # пометим выбранный, даже если он не на сетке 0.01
        print(f"  {thr:5.2f} {f:7.3f} {nfp:5d} {r:7.3f}{mark}")
    fc, rc = fpr(p, y, chosen), rec(p, y, chosen)
    print(f"  -> выбран порог {chosen:.3f}: VAL FPR {fc:.3f} ({int((p[y==0]>=chosen).sum())} ложн.), VAL recall {rc:.3f}")
    return chosen


# ============================ ОТБОР ПОРОГОВ НА VAL ============================
print("=" * 72)
print("  ШАГ 1 — отбор порогов ТОЛЬКО на VAL (цель: FPR <= %.1f%%)" % (TARGET_VAL_FPR * 100))
print("=" * 72)
va = load("data/v2_val.jsonl")
yv = np.array([r["label"] for r in va])
tv = [r["text"] for r in va]
p2v = TF.proba(tv)
p1v = v1p(tv)
pev = ens(tv, p1v, p2v)

T2 = val_table("v2 (трансформер)", p2v, yv, TARGET_VAL_FPR)
TE = val_table("ансамбль", pev, yv, TARGET_VAL_FPR)
print(f"\n  ИТОГ порогов: v2: {OLD_THR} -> {T2:.3f}   ансамбль -> {TE:.3f}")


# ===================== ЗАМЕР НА TEST + HOLD-OUT (ОДИН РАЗ) ====================
def eval_set(path):
    rows = load(path)
    t = [r["text"] for r in rows]
    y = np.array([r["label"] for r in rows])
    return t, y, v1p(t), TF.proba(t)


print("\n" + "=" * 72)
print("  ШАГ 2 — ФИНАЛЬНЫЙ замер на TEST + hold-out (один раз, по выбранным порогам)")
print("=" * 72)

tt, yt, p1t, p2t = eval_set("data/test.jsonl")
pet = ens(tt, p1t, p2t)

print("\n  --- TEST (свои генераторы, >=200): инвариант FPR <= 3% ---")
print(f"  {'модель':10s} {'порог':>6} {'ROC-AUC':>8} {'FPR':>7} {'n_fp':>5} {'recall ИИ':>10}")
for tag, p, thr in [("v2 СТАР", p2t, OLD_THR), ("v2 НОВ", p2t, T2),
                    ("ens НОВ", pet, TE)]:
    nfp = int((p[yt == 0] >= thr).sum())
    flag = "" if fpr(p, yt, thr) <= 0.03 else "  ⚠ >3%"
    print(f"  {tag:10s} {thr:6.3f} {roc_auc_score(yt,p):8.3f} {fpr(p,yt,thr):7.3f} "
          f"{nfp:5d} {rec(p,yt,thr):10.3f}{flag}")

# hold-out: невиданные генераторы (только label=1), recall == доля пойманных
print("\n  --- HOLD-OUT (невиданный ИИ, recall) — сколько генерализации осталось ---")
print(f"  {'набор':22s} {'n':>4} {'v1@0.86':>8} {'v2 СТАР':>8} {'v2 НОВ':>8} {'ens НОВ':>8}")
for name, path in [("GPT-4 (hold-out)", "data/holdout_gpt4.jsonl"),
                   ("saiga (hold-out)", "data/holdout_saiga.jsonl")]:
    t, y, p1, p2 = eval_set(path)
    pe = ens(t, p1, p2)
    r_v1 = rec(p1, y, 0.86)
    r_v2_old = rec(p2, y, OLD_THR)
    r_v2_new = rec(p2, y, T2)
    r_ens = rec(pe, y, TE)
    print(f"  {name:22s} {len(y):>4} {r_v1:8.3f} {r_v2_old:8.3f} {r_v2_new:8.3f} {r_ens:8.3f}")

# ============================ ЦЕНА УЖЕСТОЧЕНИЯ ============================
print("\n" + "=" * 72)
print("  ЦЕНА ужесточения порога v2: 0.755 -> %.3f" % T2)
print("=" * 72)
g4_t, g4_y, _, g4_p2 = eval_set("data/holdout_gpt4.jsonl")
sg_t, sg_y, _, sg_p2 = eval_set("data/holdout_saiga.jsonl")
rows_cost = [
    ("TEST FPR (людей)", fpr(p2t, yt, OLD_THR), fpr(p2t, yt, T2)),
    ("TEST recall ИИ", rec(p2t, yt, OLD_THR), rec(p2t, yt, T2)),
    ("GPT-4 recall", rec(g4_p2, g4_y, OLD_THR), rec(g4_p2, g4_y, T2)),
    ("saiga recall", rec(sg_p2, sg_y, OLD_THR), rec(sg_p2, sg_y, T2)),
]
print(f"  {'метрика':20s} {'было@0.755':>11} {'стало@%.3f' % T2:>11} {'Δ п.п.':>9}")
for name, old, new in rows_cost:
    print(f"  {name:20s} {old:11.3f} {new:11.3f} {(new-old)*100:+9.1f}")

# v1 baseline для контекста (GPT-4 был 0.13)
print(f"\n  напоминание: v1 на GPT-4 = {rec(g4_p2*0+v1p(g4_t),g4_y,0.86):.3f} (потолок признаков)")

# ============================ ЗАПИСЬ threshold_v2.json ============================
out = {
    "threshold": round(T2, 3),
    "threshold_ensemble": round(TE, 3),
    "base": "cointegrated/rubert-tiny2",
    "selection": "smallest VAL threshold with FPR<=%.3f (margin under 3%% product cap)" % TARGET_VAL_FPR,
    "val_fpr": round(fpr(p2v, yv, T2), 4),
    "val_recall": round(rec(p2v, yv, T2), 4),
    "test_auc": round(roc_auc_score(yt, p2t), 4),
    "test_fpr": round(fpr(p2t, yt, T2), 4),
    "test_recall": round(rec(p2t, yt, T2), 4),
    "holdout_gpt4_recall": round(rec(g4_p2, g4_y, T2), 4),
    "holdout_saiga_recall": round(rec(sg_p2, sg_y, T2), 4),
    "ensemble": {
        "threshold": round(TE, 3),
        "test_fpr": round(fpr(pet, yt, TE), 4),
        "test_recall": round(rec(pet, yt, TE), 4),
        "holdout_gpt4_recall": round(rec(ens(g4_t, v1p(g4_t), g4_p2), g4_y, TE), 4),
        "holdout_saiga_recall": round(rec(ens(sg_t, v1p(sg_t), sg_p2), sg_y, TE), 4),
    },
    "prev_threshold": OLD_THR,
    "prev_test_fpr": 0.0451,
}
json.dump(out, open(model_path("threshold_v2.json"), "w"), ensure_ascii=False, indent=2)
print("\n[записано] threshold_v2.json")
print(json.dumps(out, ensure_ascii=False, indent=2))
