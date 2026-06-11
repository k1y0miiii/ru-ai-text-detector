#!/usr/bin/env python3
"""
probe_base.py — ЗОНД ПОТОЛКА: подвижен ли AUC на коротком при более крупной базе,
ДО любого дообучения под детекцию.

ЧЕСТНОСТЬ ЗАМЕРА (по ТЗ заказчика). Голый AUC недообученной базы был бы низким не
из-за потолка, а из-за отсутствия обучения. Поэтому меряем ПОТОЛОК РЕПРЕЗЕНТАЦИИ
через LINEAR PROBE: база ЗАМОРОЖЕНА (никакого дообучения энкодера), сверху обучаем
ТОЛЬКО линейную голову (logistic regression) на нашем train. Это не дообучение
модели — это зонд: «сколько разделяющего сигнала на коротком несут эмбеддинги базы».
  - frozen probe — НИЖНЯЯ оценка того, что даст полное дообучение базы.
  - сравнение base-probe vs tiny2-probe честно (одинаковый протокол: pooling, голова,
    train, eval). Если крупная база при ТОМ ЖЕ протоколе даёт выше short-AUC —
    потолок репрезентации двигается, полное дообучение базы тем более поможет.

OUT-OF-SAMPLE строго: голова учится на data/v2_train.jsonl; AUC и rec@FPR3% —
на НЕВИДАННЫХ генераторах (holdout gpt4+saiga) + test-human (нет в train).
Сетка длин — как в части А (урезание до первых N слов).

Референс: v2-deployed = дообученный tiny2 (transformer_score) — что у нас сейчас.
"""

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from aidetector import transformer_score as TF
from aidetector.document_scan import _WORD_RE

MAXLEN = 256
LENGTHS = [300, 150, 100, 75, 50, 40, 30, 20]

# (tag, hf_path, prefix). prefix="query: " для e5 (его штатный формат), иначе "".
# e5-small (118M, 12 слоёв, 384h, sentence-tuned) = ~4x ёмкости tiny2 (29M, 3 слоя,
# 312h). Полную ruBert-base (711МБ) не удалось скачать в этой сети — взяли e5-small
# (470МБ) как «более крупную русскоязычную (multilingual) sentence-базу».
CANDIDATES = [
    ("tiny2-probe (29M, 3L, 312h)", "cointegrated/rubert-tiny2", ""),
    ("e5-small-probe (118M, 12L, 384h)", "bases/e5-small", "query: "),
]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def trunc(text, n):
    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if len(spans) <= n:
        return text.strip()
    return text[:spans[n - 1][1]].strip()


@torch.no_grad()
def embed(model, tok, texts, prefix="", bs=32):
    """Masked mean-pooling последнего слоя -> вектор на текст (замороженная база)."""
    vecs = []
    for i in range(0, len(texts), bs):
        chunk = [prefix + t for t in texts[i:i + bs]]
        enc = tok(chunk, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        out = model(**enc).last_hidden_state                       # (B,T,H)
        mask = enc["attention_mask"].unsqueeze(-1).float()         # (B,T,1)
        summed = (out * mask).sum(1)
        cnt = mask.sum(1).clamp(min=1e-9)
        vecs.append((summed / cnt).cpu().numpy())
    return np.vstack(vecs) if vecs else np.zeros((0, model.config.hidden_size))


def fpr(p, t):
    return float((np.asarray(p) >= t).mean()) if len(p) else 0.0


def eval_probe(score_of, tag):
    """score_of(texts)->p(ИИ). Печатает сетку длина×AUC×rec@FPR3% out-of-sample."""
    test_h = [r["text"] for r in load("data/test.jsonl") if r["label"] == 0]
    gpt4 = [r["text"] for r in load("data/holdout_gpt4.jsonl")]
    saiga = [r["text"] for r in load("data/holdout_saiga.jsonl")]
    print(f"\n  {'N':>5} {'med_w':>6} | {'AUC_all':>8} {'AUC_gpt4':>9} {'AUC_saiga':>10} | "
          f"{'rec@FPR3%':>9} {'rec_gpt4':>8}")
    rows = []
    for N in LENGTHS:
        th = score_of([trunc(t, N) for t in test_h])
        pg = score_of([trunc(t, N) for t in gpt4])
        ps = score_of([trunc(t, N) for t in saiga])
        pa = np.concatenate([pg, ps])
        y = np.r_[np.zeros(len(th)), np.ones(len(pa))]
        auc = roc_auc_score(y, np.r_[th, pa])
        aucg = roc_auc_score(np.r_[np.zeros(len(th)), np.ones(len(pg))], np.r_[th, pg])
        aucs = roc_auc_score(np.r_[np.zeros(len(th)), np.ones(len(ps))], np.r_[th, ps])
        tcal = float(np.quantile(th, 0.97))
        rec = fpr(pa, tcal); recg = fpr(pg, tcal)
        medw = int(np.median([len(_WORD_RE.findall(trunc(t, N))) for t in (test_h + gpt4 + saiga)]))
        print(f"  {N:>5} {medw:>6} | {auc:>8.3f} {aucg:>9.3f} {aucs:>10.3f} | {rec:>9.3f} {recg:>8.3f}")
        rows.append(dict(N=N, med_w=medw, auc_all=auc, auc_gpt4=aucg, auc_saiga=aucs,
                         rec_at_fpr3=rec, rec_gpt4=recg))
    return rows


def probe_model(tag, path, prefix, train_texts, train_y):
    tok = AutoTokenizer.from_pretrained(path)
    enc = AutoModel.from_pretrained(path).eval()
    Xtr = embed(enc, tok, train_texts, prefix)
    scaler = StandardScaler().fit(Xtr)
    head = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    head.fit(scaler.transform(Xtr), train_y)

    def score_of(texts):
        E = embed(enc, tok, texts, prefix)
        return head.predict_proba(scaler.transform(E))[:, 1]
    return eval_probe(score_of, tag)


def main():
    tr = load("data/v2_train.jsonl")
    train_texts = [r["text"] for r in tr]
    train_y = np.array([r["label"] for r in tr])
    print(f"linear probe: train(голова)={len(train_texts)} | eval OUT-OF-SAMPLE "
          f"(holdout gpt4/saiga + test-human)")

    out = {}
    # референс: деплойный дообученный tiny2 (AUC инвариантен к калибровке)
    print("\n" + "=" * 78 + f"\n  РЕФЕРЕНС: v2-deployed (дообученный tiny2, transformer_score)\n" + "=" * 78)
    out["v2-deployed (fine-tuned tiny2)"] = eval_probe(lambda ts: np.asarray(TF.proba(ts)), "v2")

    for tag, path, prefix in CANDIDATES:
        print("\n" + "=" * 78 + f"\n  {tag}  [{path}]  (frozen + linear head)\n" + "=" * 78)
        try:
            out[tag] = probe_model(tag, path, prefix, train_texts, train_y)
        except Exception as e:
            print(f"  [пропуск] не загрузилась: {type(e).__name__}: {str(e)[:120]}")

    json.dump(out, open("data/partD_probe.json", "w"), ensure_ascii=False, indent=2)
    print("\n[сохранено] data/partD_probe.json")

    # сводка-дельта на коротком: base-probe vs tiny2-probe и vs v2-deployed
    print("\n" + "=" * 78 + "\n  СВОДКА: AUC_all на коротком (двигается ли потолок репрезентации?)\n" + "=" * 78)
    keys = [k for k in out]
    bylen = {k: {r["N"]: r["auc_all"] for r in out[k]} for k in keys}
    print(f"  {'модель':34s} | " + " ".join(f"{N:>6}" for N in [75, 50, 40, 30, 20]))
    for k in keys:
        print(f"  {k:34s} | " + " ".join(f"{bylen[k][N]:>6.3f}" for N in [75, 50, 40, 30, 20]))


if __name__ == "__main__":
    main()
