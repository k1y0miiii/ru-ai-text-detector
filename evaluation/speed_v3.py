#!/usr/bin/env python3
"""
speed_v3.py — инференс v3 (ruBert-base) vs v2 (tiny2) НА CPU. Скорость — обязательное
требование ТЗ: прод (TUI/скан) должен остаться отзывчивым. Если v3 слишком медленный —
сигнал к ансамблю «base на коротких + tiny2 на длинных».

Меряем на CPU (как реальный self-hosted прод без GPU): латентность одиночного текста
и пропускную способность батча на реалистичном окне (~150 слов).
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from aidetector.paths import model_path

MAXLEN = 256
torch.set_num_threads(os.cpu_count() or 4)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def make(path):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).to("cpu").eval()

    @torch.no_grad()
    def score(texts, bs):
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt")
            torch.softmax(m(**enc).logits, 1)
    return tok, m, score


def bench(score, texts, bs, reps=3):
    score(texts[:bs], bs)  # warmup
    best = 1e9
    for _ in range(reps):
        t0 = time.time()
        score(texts, bs)
        best = min(best, time.time() - t0)
    return best


def main():
    # реалистичные окна ~150 слов из test
    rows = load("data/test.jsonl")
    pool = [r["text"] for r in rows if len(r["text"]) >= 300][:64] or [r["text"] for r in rows][:64]
    print(f"[speed] CPU, {torch.get_num_threads()} threads | {len(pool)} текстов ~окно\n")

    out = {}
    for tag, path in [("v2 (tiny2)", model_path("v2_model")), ("v3 (ruBert-base)", model_path("v3_model"))]:
        if not os.path.exists(path):
            print(f"  [пропуск] нет {path}")
            continue
        _, m, score = make(path)
        nparams = sum(p.numel() for p in m.parameters())
        t_single = bench(score, pool[:1], 1, reps=5)
        t_batch = bench(score, pool, 32, reps=3)
        per_text_batch = t_batch / len(pool)
        out[tag] = {"params_M": nparams / 1e6, "ms_single": t_single * 1000,
                    "ms_per_text_batch32": per_text_batch * 1000,
                    "texts_per_sec_batch32": len(pool) / t_batch}
        print(f"  {tag:20s} params={nparams/1e6:>4.0f}M | одиночный {t_single*1000:>6.1f} мс | "
              f"батч32 {per_text_batch*1000:>5.1f} мс/текст ({len(pool)/t_batch:>5.1f} тек/с)")

    if "v2 (tiny2)" in out and "v3 (ruBert-base)" in out:
        r = out["v3 (ruBert-base)"]["ms_per_text_batch32"] / out["v2 (tiny2)"]["ms_per_text_batch32"]
        rs = out["v3 (ruBert-base)"]["ms_single"] / out["v2 (tiny2)"]["ms_single"]
        print(f"\n  v3 медленнее v2 на CPU: батч ×{r:.1f}, одиночный ×{rs:.1f}")
        # прикидка: типичный скан-документ ~10-20 окон
        ms_doc_v3 = out["v3 (ruBert-base)"]["ms_per_text_batch32"] * 15
        ms_doc_v2 = out["v2 (tiny2)"]["ms_per_text_batch32"] * 15
        print(f"  прикидка скана документа (~15 окон): v2 {ms_doc_v2:.0f} мс vs v3 {ms_doc_v3:.0f} мс")
        if ms_doc_v3 > 3000:
            print("  [!] v3-скан >3с/док на CPU — рекомендуется АНСАМБЛЬ (v3 коротким / tiny2 длинным) "
                  "или ONNX/квантизация.")
        else:
            print("  v3-скан в пределах отзывчивости (<3с/док на CPU).")

    json.dump(out, open("data/v3_speed.json", "w"), ensure_ascii=False, indent=2)
    print("\n[сохранено] data/v3_speed.json")


if __name__ == "__main__":
    main()
