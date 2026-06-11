#!/usr/bin/env python3
"""
evaluate_v3.py — ПРИЁМКА v3 против v2. Метрика — recall@FPR3% OUT-OF-SAMPLE,
НЕ AUC и НЕ recall@фикс.порог (анти-маскировка).

Критерий приёмки (ТЗ):
  - rec@FPR3% на 40 словах: v3 >= v2 + 10 п.п. (на невиданных holdout gpt4/saiga);
  - FPR <= 3% на КАЖДОЙ длине (realized на test-human, порог калиброван на VAL-human);
  - длинные (>=200 симв, свои генераторы) и общий test-FPR — не хуже v2 (~0.023);
  - прирост < 5 п.п. в бюджете FPR -> откат, фиксируем инфо-границу ~40 слов.

Методология (как в части C): поверхность РЕШЕНИЯ — сырой softmax; порог на длину
калибруется на VAL-human (квантиль 0.975 -> FPR~2.5%, запас под 3%); recall меряется
на holdout-AI, realized FPR — на test-human (нет в train/val). Для обеих моделей
ОДИН протокол.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aidetector.document_scan import _WORD_RE
from aidetector.paths import model_path

LENGTHS = [300, 150, 100, 75, 50, 40, 30, 20]
MAXLEN = 256


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def trunc(t, n):
    sp = [(m.start(), m.end()) for m in _WORD_RE.finditer(t)]
    return t.strip() if len(sp) <= n else t[:sp[n - 1][1]].strip()


def device_pick():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_scorer(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

    @torch.no_grad()
    def score(texts, bs=64):
        ps = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to(device)
            ps.append(torch.softmax(m(**enc).logits, 1)[:, 1].float().cpu().numpy())
        return np.concatenate(ps) if ps else np.array([])
    return score


def sweep(score, val_h, test_h, gpt4, saiga):
    """Per-length: порог из VAL-human (FPR<=2.5%), recall на holdout-AI, realized test-FPR."""
    rows = []
    for N in LENGTHS:
        vh = score([trunc(t, N) for t in val_h])
        th = score([trunc(t, N) for t in test_h])
        pg = score([trunc(t, N) for t in gpt4])
        ps = score([trunc(t, N) for t in saiga])
        pa = np.concatenate([pg, ps])
        t_L = float(np.quantile(vh, 0.975))
        rec = float((pa >= t_L).mean())
        recg = float((pg >= t_L).mean())
        fprt = float((th >= t_L).mean())
        auc = roc_auc_score(np.r_[np.zeros(len(th)), np.ones(len(pa))], np.r_[th, pa])
        aucg = roc_auc_score(np.r_[np.zeros(len(th)), np.ones(len(pg))], np.r_[th, pg])
        rows.append(dict(N=N, t=t_L, rec=rec, recg=recg, fpr=fprt, auc=auc, aucg=aucg))
    return rows


def global_long(score, val_h, test_h, test_ai_long):
    """Глобальный порог (полнодлинный VAL, FPR<=2.5%); общий test-FPR и recall на длинных своих."""
    tg = float(np.quantile(score(val_h), 0.975))
    overall_fpr = float((score(test_h) >= tg).mean())
    long_rec = float((score(test_ai_long) >= tg).mean())
    return tg, overall_fpr, long_rec


def main():
    dev = device_pick()
    print(f"[device] {dev}\n")
    val_h = [r["text"] for r in load("data/v2_val.jsonl") if r["label"] == 0]
    test_h = [r["text"] for r in load("data/test.jsonl") if r["label"] == 0]
    test_ai_long = [r["text"] for r in load("data/test.jsonl")
                    if r["label"] == 1 and len(r["text"]) >= 200]
    gpt4 = [r["text"] for r in load("data/holdout_gpt4.jsonl")]
    saiga = [r["text"] for r in load("data/holdout_saiga.jsonl")]

    models = [("v2", model_path("v2_model")), ("v3", model_path("v3_model"))]
    res = {}
    for tag, path in models:
        if not os.path.exists(path):
            print(f"[пропуск] нет {path}")
            continue
        sc = make_scorer(path, dev)
        res[tag] = {"sweep": sweep(sc, val_h, test_h, gpt4, saiga)}
        tg, ofpr, lrec = global_long(sc, val_h, test_h, test_ai_long)
        res[tag]["global"] = {"thr": tg, "overall_test_fpr": ofpr, "long_recall": lrec}

    # --- таблица rec@FPR3% по длинам: v2 vs v3 ---
    print("=" * 88)
    print("  rec@FPR3% (recall на невиданных gpt4+saiga при FPR<=3%, OUT-OF-SAMPLE) — v2 vs v3")
    print("=" * 88)
    print(f"  {'N':>5} | {'v2 rec':>7} {'v2 fpr':>7} {'v2 recG':>8} | "
          f"{'v3 rec':>7} {'v3 fpr':>7} {'v3 recG':>8} | {'Δrec':>6} {'ΔrecG':>6}")
    if "v2" in res and "v3" in res:
        for r2, r3 in zip(res["v2"]["sweep"], res["v3"]["sweep"]):
            f2 = "" if r2["fpr"] <= 0.03 + 1e-9 else "!"
            f3 = "" if r3["fpr"] <= 0.03 + 1e-9 else "!"
            star = "  <<40сл" if r3["N"] == 40 else ""
            print(f"  {r3['N']:>5} | {r2['rec']:>7.3f} {r2['fpr']:>6.3f}{f2:1s} {r2['recg']:>8.3f} | "
                  f"{r3['rec']:>7.3f} {r3['fpr']:>6.3f}{f3:1s} {r3['recg']:>8.3f} | "
                  f"{r3['rec']-r2['rec']:>+6.3f} {r3['recg']-r2['recg']:>+6.3f}{star}")

    print("\n  AUC_all / AUC_gpt4 по длинам (справочно, НЕ критерий):")
    print(f"  {'N':>5} | {'v2 AUC':>7} {'v3 AUC':>7} | {'v2 AUCg':>8} {'v3 AUCg':>8}")
    if "v2" in res and "v3" in res:
        for r2, r3 in zip(res["v2"]["sweep"], res["v3"]["sweep"]):
            print(f"  {r3['N']:>5} | {r2['auc']:>7.3f} {r3['auc']:>7.3f} | {r2['aucg']:>8.3f} {r3['aucg']:>8.3f}")

    print("\n  Глобальный порог (полнодлинный VAL FPR<=2.5%): «длинные/общий не хуже v2»")
    for tag in ("v2", "v3"):
        if tag in res:
            g = res[tag]["global"]
            print(f"   {tag}: общий test-FPR={g['overall_test_fpr']:.3f}  "
                  f"recall длинные(>=200симв, свои)={g['long_recall']:.3f}  (порог {g['thr']:.3f})")

    # --- вердикт приёмки ---
    if "v2" in res and "v3" in res:
        r40_2 = next(r for r in res["v2"]["sweep"] if r["N"] == 40)
        r40_3 = next(r for r in res["v3"]["sweep"] if r["N"] == 40)
        d40 = r40_3["rec"] - r40_2["rec"]
        d40g = r40_3["recg"] - r40_2["recg"]
        all_fpr_ok = all(r["fpr"] <= 0.03 + 1e-9 for r in res["v3"]["sweep"])
        long_ok = res["v3"]["global"]["overall_test_fpr"] <= res["v2"]["global"]["overall_test_fpr"] + 0.005
        print("\n" + "=" * 88)
        print(f"  ВЕРДИКТ: Δrec@FPR3%(40сл)={d40:+.3f} (gpt4 {d40g:+.3f}) | "
              f"FPR<=3% на всех длинах: {'ДА' if all_fpr_ok else 'НЕТ'} | "
              f"общий FPR не хуже v2: {'ДА' if long_ok else 'НЕТ'}")
        if d40 >= 0.10 and all_fpr_ok and long_ok:
            print("  => ПРИЁМКА ПРОЙДЕНА (>=+10пп, FPR в бюджете). v3 оправдан.")
        elif d40 >= 0.05 and all_fpr_ok:
            print("  => ЧАСТИЧНО (+5..10пп). Обсудить: цена скорости vs выигрыш.")
        else:
            print("  => ОТКАТ: прирост <5пп в бюджете FPR -> инфо-граница ~40 слов как свойство продукта.")
        print("=" * 88)

    json.dump(res, open("data/v3_eval.json", "w"), ensure_ascii=False, indent=2)
    print("\n[сохранено] data/v3_eval.json")


if __name__ == "__main__":
    main()
