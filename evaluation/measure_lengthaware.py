#!/usr/bin/env python3
"""
measure_lengthaware.py — length-aware порог: вернуть recall на КОРОТКОМ без роста FPR.

ИДЕЯ. Единый порог v2 калиброван на смеси длин. На коротком фрагменте распределение
скоров уезжает -> единый порог становится неоптимальным. Порог, ЗАВИСЯЩИЙ от длины,
ставим под FPR<=3% ОТДЕЛЬНО на каждой длине и смотрим, вернётся ли recall.

АНТИ-МАСКИРОВКА (требование заказчика). Выигрыш засчитывается ТОЛЬКО если recall
вырос ПРИ УДЕРЖАННОМ FPR<=3%. Калибруем порог на VAL-human (квантиль 0.975 -> VAL
FPR~2.5%, запас под 3%), а FPR ПРОВЕРЯЕМ на отложенном test-human. Сравниваем не с
recall@0.90, а с recall ЕДИНОГО порога на той же поверхности при том же бюджете FPR
(length-aware vs global). Метрика — recall@FPR3%.

ДВЕ ПОВЕРХНОСТИ СКОРА:
  calibrated — изотоника TF.proba (то, что показываем как «вероятность»). У неё
               ПЛАТО на ~0.90: масса текстов слипается в потолок -> порог НЕ двигается
               (длина-инвариантно ~0.90) и при VAL->test даёт скачки FPR. Негодна для
               тонкой настройки порога.
  raw        — сырой softmax модели (до изотоники). Непрерывен, есть запас выше 0.90 ->
               на нём порог можно ставить осмысленно. Это поверхность РЕШЕНИЯ; калиброванную
               p оставляем только для ПОКАЗА пользователю.

Recall — на НЕВИДАННЫХ генераторах (holdout gpt4+saiga). Модель НЕ трогаем.
"""

import json

import numpy as np
import torch

from aidetector import transformer_score as TF
from aidetector.document_scan import _WORD_RE

LENGTHS = [300, 150, 100, 75, 50, 40, 30, 20]
VAL_Q = 0.975            # квантиль VAL-human -> VAL FPR ~2.5% (запас под 3% на test)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def trunc(text, n):
    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if len(spans) <= n:
        return text.strip()
    return text[:spans[n - 1][1]].strip()


@torch.no_grad()
def raw_proba(texts, bs=64):
    """Сырой softmax p(ИИ) (ДО изотоники) — поверхность решения для порога."""
    TF._ensure()
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = TF._t(chunk, truncation=True, max_length=TF.MAXLEN, padding=True, return_tensors="pt")
        out.append(torch.softmax(TF._m(**enc).logits, dim=1)[:, 1].numpy())
    return np.concatenate(out) if out else np.array([])


def fpr(p, t):
    return float((np.asarray(p) >= t).mean()) if len(p) else 0.0


def run_surface(name, score_fn, val_h, test_h, gpt4, saiga):
    print("\n" + "=" * 92)
    print(f"  ПОВЕРХНОСТЬ СКОРА: {name}")
    print("=" * 92)
    # глобальный порог на этой поверхности: калибруем на ПОЛНОДЛИННОМ VAL-human (FPR~2.5%)
    g_thr = float(np.quantile(score_fn(val_h), VAL_Q))
    print(f"  единый (global) порог = квантиль{VAL_Q} полнодлинного VAL-human = {g_thr:.4f}\n")
    h = (f"  {'N(cap)':>6} {'med_w':>6} | {'t_len':>7} {'val_fpr':>7} {'test_fpr':>8} "
         f"{'REC@FPR3%':>9} {'rec_gpt4':>8} | {'-- единый порог --':>18} | {'Δrec':>7}")
    print(h)
    print(f"  {'':>13} | {'':>32} (length-aware) | {'rec':>5} {'fpr_t':>6} {'rec_g':>5} |")
    print("  " + "-" * (len(h) - 2))
    rows = []
    for N in LENGTHS:
        vh = score_fn([trunc(t, N) for t in val_h])
        th = score_fn([trunc(t, N) for t in test_h])
        pg = score_fn([trunc(t, N) for t in gpt4])
        ps = score_fn([trunc(t, N) for t in saiga])
        pa = np.concatenate([pg, ps])
        med_w = int(np.median([len(_WORD_RE.findall(trunc(t, N)))
                               for t in (val_h + test_h + gpt4 + saiga)]))

        t_len = float(np.quantile(vh, VAL_Q))           # length-aware порог
        la_fpr_v, la_fpr_t = fpr(vh, t_len), fpr(th, t_len)
        la_rec, la_rec_g = fpr(pa, t_len), fpr(pg, t_len)
        g_rec, g_fpr_t, g_rec_g = fpr(pa, g_thr), fpr(th, g_thr), fpr(pg, g_thr)

        flag = "" if la_fpr_t <= 0.03 + 1e-9 else " FPR>3%!"
        print(f"  {N:>6} {med_w:>6} | {t_len:>7.4f} {la_fpr_v:>7.3f} {la_fpr_t:>8.3f} "
              f"{la_rec:>9.3f} {la_rec_g:>8.3f} | {g_rec:>5.3f} {g_fpr_t:>6.3f} {g_rec_g:>5.3f} | "
              f"{la_rec - g_rec:>+7.3f}{flag}")
        rows.append(dict(N=N, med_w=med_w, t_len=t_len, val_fpr=la_fpr_v, test_fpr=la_fpr_t,
                         recall_at_fpr3=la_rec, recall_gpt4=la_rec_g,
                         global_thr=g_thr, global_recall=g_rec, global_fpr_test=g_fpr_t,
                         global_recall_gpt4=g_rec_g))
    return rows


def main():
    val_h = [r["text"] for r in load("data/v2_val.jsonl") if r["label"] == 0]
    test_h = [r["text"] for r in load("data/test.jsonl") if r["label"] == 0]
    gpt4 = [r["text"] for r in load("data/holdout_gpt4.jsonl")]
    saiga = [r["text"] for r in load("data/holdout_saiga.jsonl")]
    print(f"калибровка VAL-human={len(val_h)}  |  оценка: test-human={len(test_h)}  "
          f"ИИ gpt4={len(gpt4)} saiga={len(saiga)}")

    out = {}
    out["calibrated"] = run_surface("calibrated (изотоника, как сейчас)",
                                    lambda ts: np.asarray(TF.proba(ts)), val_h, test_h, gpt4, saiga)
    out["raw"] = run_surface("raw softmax (поверхность решения для порога)",
                             raw_proba, val_h, test_h, gpt4, saiga)

    json.dump(out, open("data/partC_lengthaware.json", "w"), ensure_ascii=False, indent=2)
    print("\n[сохранено] data/partC_lengthaware.json")
    print("\nЧтение: Δrec = recall(length-aware) − recall(единый порог) при FPR<=3% на ОБОИХ. "
          ">0 и test_fpr<=3% = length-aware реально вернул recall. =0 = порог не помогает "
          "(длина не меняет оптимум) либо плато (calibrated).")


if __name__ == "__main__":
    main()
