#!/usr/bin/env python3
"""
train_v2.py — дообучение трансформера rubert-tiny2 (бинарный human/AI) — этап v2.

Цель v2 (после дата-эксперимента): ГЕНЕРАЛИЗАЦИЯ на невиданные генераторы, где
feature-based уперся (GPT-4 recall 0.13). GPT-4/saiga в train НЕ входят.

База: cointegrated/rubert-tiny2 (3 слоя, hidden 312 — быстрая на CPU). num_labels=2.
После обучения: изотоническая калибровка на val + порог под FPR<=3% (как у v1, чтобы
пороги были сопоставимы). Выходы отдельные — v2_model/ + v2_calibrator.joblib +
threshold_v2.json. v1/v1.1 не трогаем.
"""

import json
import time

import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from aidetector.dataset import load_jsonl
from aidetector.paths import model_path

MODEL = "cointegrated/rubert-tiny2"
MAXLEN = 256
torch.manual_seed(42)
tok = AutoTokenizer.from_pretrained(MODEL)


class DS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.enc = tok([r["text"] for r in rows], truncation=True, max_length=MAXLEN)
        self.labels = [int(r["label"]) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        d["labels"] = self.labels[i]
        return d


def _softmax_p1(logits):
    return torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]


@torch.no_grad()
def raw_proba(model, texts, bs=32):
    """Сырая p(ИИ)=softmax трансформера для списка текстов (CPU, батчами)."""
    model.to("cpu").eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        out.append(torch.softmax(model(**enc).logits, dim=1)[:, 1].numpy())
    return np.concatenate(out) if out else np.array([])


def compute_metrics(pred):
    p1 = _softmax_p1(pred.predictions)
    y = pred.label_ids
    return {"auc": roc_auc_score(y, p1), "acc": float(((p1 >= 0.5) == y).mean())}


def fpr_recall(y, p, thr):
    pred = (np.asarray(p) >= thr).astype(int)
    return (float(pred[y == 0].mean()) if (y == 0).any() else 0.0,
            float(pred[y == 1].mean()) if (y == 1).any() else 0.0)


def main():
    tr = load_jsonl("data/v2_train.jsonl")
    va = load_jsonl("data/v2_val.jsonl")
    print(f"train={len(tr)} val={len(va)}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    args = TrainingArguments(
        output_dir="v2_ckpt", num_train_epochs=4,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.1,
        evaluation_strategy="epoch", save_strategy="no", logging_steps=50,
        report_to="none", seed=42, dataloader_num_workers=0,
        use_cpu=True)   # CPU как в features.py; MPS на Mac местами падает на инференсе
    trainer = Trainer(model=model, args=args, train_dataset=DS(tr), eval_dataset=DS(va),
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)

    t0 = time.time()
    trainer.train()
    print(f"\n[time] обучение заняло {(time.time()-t0)/60:.1f} мин")

    # калибровка + порог на VAL (как у v1) ----
    yval = np.array([int(r["label"]) for r in va])
    pval_raw = raw_proba(model, [r["text"] for r in va])
    iso = IsotonicRegression(out_of_bounds="clip").fit(pval_raw, yval)
    pval_cal = iso.predict(pval_raw)

    grid = np.round(np.arange(0.05, 0.991, 0.005), 3)
    chosen = next((float(t) for t in grid if fpr_recall(yval, pval_cal, t)[0] <= 0.03),
                  float(grid[-1]))
    fv, rv = fpr_recall(yval, pval_cal, chosen)
    print(f"[порог] T={chosen:.3f} (VAL калибр.: FPR={fv:.3f}<=0.03, recall={rv:.3f})")

    # быстрый замер на TEST (свои генераторы) ----
    te = load_jsonl("data/test.jsonl")
    yte = np.array([int(r["label"]) for r in te])
    pte_cal = iso.predict(raw_proba(model, [r["text"] for r in te]))
    auc = roc_auc_score(yte, pte_cal)
    ft, rt = fpr_recall(yte, pte_cal, chosen)
    print(f"[TEST] ROC-AUC {auc:.3f} | FPR {ft:.3f} | recall ИИ {rt:.3f}")

    model.save_pretrained(model_path("v2_model")); tok.save_pretrained(model_path("v2_model"))
    joblib.dump(iso, model_path("v2_calibrator.joblib"))
    json.dump({"threshold": chosen, "base": MODEL, "test_auc": auc,
               "test_fpr": ft, "test_recall": rt}, open(model_path("threshold_v2.json"), "w"), indent=2)
    print("Сохранено: v2_model/ + v2_calibrator.joblib + threshold_v2.json")


if __name__ == "__main__":
    main()
