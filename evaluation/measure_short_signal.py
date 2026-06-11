#!/usr/bin/env python3
"""
measure_short_signal.py — ЧАСТЬ А: есть ли у v2 (трансформера) сигнал на КОРОТКОМ
фрагменте, или короткий текст упирается в информационный предел.

ЗАЧЕМ. Прежде чем дообучать v2 под короткие тексты, надо проверить ГИПОТЕЗУ:
короткий текст плохо ловится потому, что (а) на нём физически НЕТ сигнала
(информационный предел — дообучение не поможет), или (б) сигнал есть, но текущий
v2 его не берёт (тогда дообучение осмысленно). Меряем разделимость классов
человек/ИИ как функцию длины фрагмента.

МЕТОД — урезание (truncation). ВАЖНАЯ ОГОВОРКА про данные: hold-out тексты
НЕВИДАННЫХ генераторов (GPT-4, saiga) сами по себе КОРОТКИЕ (медиана ~50 слов,
максимум 115/223). Длинных (300/150 слов) фрагментов невиданного ИИ в природе
нашего корпуса НЕТ — нарезать «кусок 300 слов» физически не из чего. Поэтому
вместо нарезки длинного на короткое мы УРЕЗАЕМ каждый фрагмент до первых N слов
и смотрим, как падает разделимость. В колонке med_w показана РЕАЛЬНАЯ медианная
длина при данном пороге N — видно, с какого N урезание вообще начинает «кусать»
(выше медианы оно почти не меняет текст). Действие — в диапазоне 75->40->20 слов.

Ничего не дообучаем и не переобучаем — только инференс v2 (transformer_score) и
метрики разделимости (ROC-AUC, overlap-коэффициент).
"""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from aidetector import transformer_score as TF
from aidetector.document_scan import _WORD_RE  # та же дефиниция «слова», что в скане

# Урезаем до этих длин (слов). 300/150 для корпуса почти не «кусают» (тексты
# короче) — оставлены для запрошенной сетки и как «полный текст» baseline.
LENGTHS = [300, 150, 100, 75, 50, 40, 30, 20]

THR = TF.threshold()  # боевой порог v2 (0.90) — для recall@thr / fpr@thr контекста


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def truncate_words(text: str, n: int) -> str:
    """Первые n «слов» текста (слово = токен с буквой), с сохранением подстроки.
    Текст короче n слов возвращается целиком."""
    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if len(spans) <= n:
        return text.strip()
    end = spans[n - 1][1]
    return text[:end].strip()


def wcount(text: str) -> int:
    return len(_WORD_RE.findall(text))


def overlap_coeff(p_h: np.ndarray, p_a: np.ndarray, bins: int = 20) -> float:
    """Overlapping coefficient двух распределений p(ИИ) на [0,1].
    1.0 = распределения совпадают (классы НЕразделимы), 0.0 = не пересекаются."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    hh, _ = np.histogram(p_h, bins=edges)
    ha, _ = np.histogram(p_a, bins=edges)
    hh = hh / max(hh.sum(), 1)
    ha = ha / max(ha.sum(), 1)
    return float(np.minimum(hh, ha).sum())


def main():
    # --- данные: ИИ (невиданные генераторы) + человек (hold-out test) ---
    gpt4 = [r["text"] for r in load("data/holdout_gpt4.jsonl")]   # ИИ, label 1
    saiga = [r["text"] for r in load("data/holdout_saiga.jsonl")]  # ИИ, label 1
    human = [r["text"] for r in load("data/test.jsonl") if r["label"] == 0]  # человек

    print(f"источники: human={len(human)}  gpt4={len(gpt4)}  saiga={len(saiga)}")
    print(f"порог v2 = {THR:.2f}\n")
    print("сетка по длине (урезание до первых N слов; med_w = реальная медиана слов "
          "при этом пороге)\n")

    header = (f"{'N(cap)':>6} {'med_w':>6} {'min_w':>6} | {'AUC_all':>8} {'AUC_gpt4':>9} "
              f"{'AUC_saiga':>10} | {'overlap':>8} {'p_hum':>6} {'p_ai':>6} | "
              f"{'rec@.9':>7} {'fpr@.9':>7} | {'t_cal':>6} {'rec@3%':>7}")
    print(header)
    print("-" * len(header))

    rows_out = []
    for N in LENGTHS:
        h = [truncate_words(t, N) for t in human]
        g = [truncate_words(t, N) for t in gpt4]
        s = [truncate_words(t, N) for t in saiga]

        ph = TF.proba(h)
        pg = TF.proba(g)
        ps = TF.proba(s)
        pa = np.concatenate([pg, ps])  # все ИИ

        y_all = np.r_[np.zeros(len(ph)), np.ones(len(pa))]
        p_all = np.r_[ph, pa]
        auc_all = roc_auc_score(y_all, p_all)
        auc_g = roc_auc_score(np.r_[np.zeros(len(ph)), np.ones(len(pg))], np.r_[ph, pg])
        auc_s = roc_auc_score(np.r_[np.zeros(len(ph)), np.ones(len(ps))], np.r_[ph, ps])

        ovl = overlap_coeff(ph, pa)
        # реальная длина после урезания (медиана/мин по ВСЕМ фрагментам)
        all_w = [wcount(x) for x in (h + g + s)]
        med_w = int(np.median(all_w))
        min_w = int(np.min(all_w))

        rec = float((pa >= THR).mean())   # recall ИИ при боевом пороге
        fpr = float((ph >= THR).mean())   # FPR на человеке при боевом пороге

        # порог, ПЕРЕсобранный под эту длину на FPR<=3% (in-sample, оценка ПОТОЛКА):
        # отделяет «информационный предел» (recall падает даже при честной калибровке)
        # от «артефакта порога» (recall падает только из-за фиксированного 0.90).
        t_cal = float(np.quantile(ph, 0.97))
        rec_cal = float((pa >= t_cal).mean())

        print(f"{N:>6} {med_w:>6} {min_w:>6} | {auc_all:>8.3f} {auc_g:>9.3f} {auc_s:>10.3f} | "
              f"{ovl:>8.3f} {np.median(ph):>6.3f} {np.median(pa):>6.3f} | "
              f"{rec:>7.3f} {fpr:>7.3f} | {t_cal:>6.3f} {rec_cal:>7.3f}")
        rows_out.append(dict(N=N, med_w=med_w, min_w=min_w, auc_all=auc_all,
                             auc_gpt4=auc_g, auc_saiga=auc_s, overlap=ovl,
                             med_p_human=float(np.median(ph)), med_p_ai=float(np.median(pa)),
                             recall_ai=rec, fpr_human=fpr, t_cal=t_cal, recall_at_fpr3=rec_cal))

    json.dump(rows_out, open("data/partA_short_signal.json", "w"), ensure_ascii=False, indent=2)
    print("\n[сохранено] data/partA_short_signal.json")
    print("\nЧтение: AUC_all падает к 0.5 и overlap растёт к 1.0 = классы перестали "
          "разделяться (информационный предел). Если AUC держится высоко даже на 40-20 "
          "словах — сигнал на коротком ЕСТЬ.")


if __name__ == "__main__":
    main()
