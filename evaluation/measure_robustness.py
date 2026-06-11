#!/usr/bin/env python3
"""
measure_robustness.py — батч-замер устойчивости детектора к ЛЁГКОМУ редактированию.

Прогоняем каждый текст из data/robustness_raw.jsonl через детектор ДО и ПОСЛЕ
фиксированных правок perturb.py (которые НЕ видят детектор) и считаем падение
recall. Замена выводу n=1: число «recall падает на N п.п.» на n=много.

Детектор — ровно тот же, что в проде: features.extract + model.joblib + порог из
threshold_v1.json. Модель/порог/features НЕ трогаем, только читаем.
"""

import json
import statistics
import sys
from collections import defaultdict

import joblib

from aidetector.features import extract, to_vector
from aidetector.perturb import perturb
from aidetector.paths import model_path

MODEL_PATH = model_path("model.joblib")
THRESHOLD_PATH = model_path("threshold_v1.json")
DEFAULT_THRESHOLD = 0.86
BATCH = "data/robustness_raw.jsonl"
RESULTS_JSON = "data/robustness_results.json"


def load_threshold() -> float:
    try:
        with open(THRESHOLD_PATH, encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return DEFAULT_THRESHOLD


def histogram(deltas: list[float]) -> str:
    """ASCII-гистограмма Δp = p_после − p_до. Δp<0 — текст стал «человечнее»."""
    edges = [-1.0, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 1.0]
    counts = [0] * (len(edges) - 1)
    for d in deltas:
        for i in range(len(edges) - 1):
            if edges[i] <= d < edges[i + 1]:
                counts[i] += 1
                break
    mx = max(counts) or 1
    lines = []
    for i in range(len(edges) - 1):
        lines.append(f"  [{edges[i]:+.2f}, {edges[i+1]:+.2f})  {counts[i]:3d} "
                     + "█" * round(counts[i] / mx * 40))
    return "\n".join(lines)


def recall(ps: list[float], thr: float) -> float:
    return sum(p >= thr for p in ps) / len(ps) if ps else 0.0


def main():
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    thr = load_threshold()
    clf = joblib.load(MODEL_PATH)
    batch = [json.loads(l) for l in open(BATCH, encoding="utf-8")]
    if n_arg:
        batch = batch[:n_arg]
    N = len(batch)
    print(f"[i] батч: {N} ИИ-текстов, порог {thr:.2f}\n")

    def p_ai(t: str) -> float:
        return float(clf.predict_proba([to_vector(extract(t))])[0][1])

    res = []
    for k, r in enumerate(batch, 1):
        pb = p_ai(r["text"])
        pa = p_ai(perturb(r["text"]))
        res.append({"bucket": r["bucket"], "generator": r["generator"],
                    "p_before": pb, "p_after": pa})
        if k % 25 == 0 or k == N:
            print(f"  ...{k}/{N}")

    pb = [x["p_before"] for x in res]
    pa = [x["p_after"] for x in res]
    deltas = [a - b for a, b in zip(pa, pb)]
    rec_b, rec_a = recall(pb, thr), recall(pa, thr)
    drop = (rec_b - rec_a) * 100
    flipped = sum(1 for b, a in zip(pb, pa) if b >= thr and a < thr)
    gained = sum(1 for b, a in zip(pb, pa) if b < thr and a >= thr)

    print("\n" + "=" * 66)
    print(f"  УСТОЙЧИВОСТЬ К ЛЁГКОМУ РЕДАКТИРОВАНИЮ (n={N}, порог {thr:.2f})")
    print("=" * 66)
    print(f"  recall ИИ (сырой):        {rec_b:.3f}  ({sum(p>=thr for p in pb)}/{N})")
    print(f"  recall ИИ (после правок): {rec_a:.3f}  ({sum(p>=thr for p in pa)}/{N})")
    print(f"  >>> ПАДЕНИЕ recall:       {drop:.1f} п.п. <<<")
    print(f"  перешло «подозрительно»→«человек»: {flipped}/{N}   (обратно: {gained})")
    print(f"  сдвиг p: median Δp={statistics.median(deltas):+.3f}  "
          f"mean Δp={statistics.mean(deltas):+.3f}  "
          f"(median p: {statistics.median(pb):.3f} -> {statistics.median(pa):.3f})")
    print("\n  Гистограмма Δp (p_после − p_до):")
    print(histogram(deltas))

    # --- A (свои генераторы) vs B (чужой/невиданный ИИ) ---
    print("\n  recall по корзинам (сырой -> после правок):")
    for bucket in ("A", "B"):
        xs = [x for x in res if x["bucket"] == bucket]
        if xs:
            rb = recall([x["p_before"] for x in xs], thr)
            ra = recall([x["p_after"] for x in xs], thr)
            tag = "свои генераторы" if bucket == "A" else "чужой ИИ (saiga/gpt4)"
            print(f"    {bucket} [{tag:22s}] n={len(xs):3d}: {rb:.3f} -> {ra:.3f}  ({(rb-ra)*100:+.1f} п.п.)")

    print("\n  recall по генераторам (сырой -> после):")
    by = defaultdict(list)
    for x in res:
        by[x["generator"]].append(x)
    for g, xs in sorted(by.items()):
        rb = recall([x["p_before"] for x in xs], thr)
        ra = recall([x["p_after"] for x in xs], thr)
        seen = "(в обучении)" if g in ("chatgpt", "qwen3b", "coat_machine") else "(НЕ в обучении)"
        print(f"    {g:14s} {seen:16s} n={len(xs):3d}: {rb:.3f} -> {ra:.3f}")

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"n": N, "threshold": thr, "recall_before": rec_b,
                   "recall_after": rec_a, "drop_pp": drop,
                   "flipped_susp_to_human": flipped,
                   "median_delta_p": statistics.median(deltas),
                   "per_text": res}, f, ensure_ascii=False, indent=2)
    print(f"\n[i] сырые результаты -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
