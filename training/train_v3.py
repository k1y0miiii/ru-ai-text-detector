#!/usr/bin/env python3
"""
train_v3.py — дообучение БОЛЬШЕЙ русской энкодер-базы (ruBert-base, план Б e5-base)
под детекцию, ПРИЦЕЛЬНО на короткие фрагменты. Это v3. v1/v2 НЕ трогаем.

ТЗ (утверждено):
  - база: ai-forever/ruBert-base (приоритет), иначе multilingual-e5-base.
  - аугментация урезанием: в train добавляем версии 75/50/40/30/20 слов (длинные
    НЕ выбрасываем) — учим модель явно на коротком, где цель.
  - полный fine-tune энкодера (не frozen — frozen был лишь зондом потолка).
  - выбор чекпойнта по val short-AUC (40/30/20 слов) на v2_val.
  - железо M3 Max -> device="mps"; при падении op на MPS — PYTORCH_ENABLE_MPS_FALLBACK=1,
    НО громко предупреждаем, молча на CPU не сваливаемся. Печатаем фактическое время эпохи.
  Калибровка/пороги/приёмка — отдельными скриптами (calibrate_v3 / evaluate_v3).
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from aidetector.document_scan import _WORD_RE
from aidetector.paths import model_path

MAXLEN = 192     # фикс. длина токенов; корпус короткий (медиана ~35 слов) -> 192 покрывает
AUG_LENGTHS = [75, 50, 40, 30, 20]   # урезания для аугментации
SEL_LENGTHS = [40, 30, 20]           # по чему выбираем лучший чекпойнт (короткое)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def wc(t):
    return len(_WORD_RE.findall(t))


def trunc(t, n):
    sp = [(m.start(), m.end()) for m in _WORD_RE.finditer(t)]
    return t.strip() if len(sp) <= n else t[:sp[n - 1][1]].strip()


def build_aug(rows):
    """Полные тексты + их урезания до AUG_LENGTHS (где текст длиннее). Дедуп по (text,label)."""
    seen, out = set(), []
    for r in rows:
        t, y = r["text"], r["label"]
        variants = [t] + [trunc(t, n) for n in AUG_LENGTHS if wc(t) > n]
        for v in variants:
            k = (v, y)
            if v.strip() and k not in seen:
                seen.add(k)
                out.append((v, y))
    return out


class DS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(tok):
    def collate(batch):
        texts = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        # ФИКСИРОВАННАЯ длина: на MPS динамический padding плодит граф на КАЖДУЮ длину
        # последовательности -> кэш графов раздувается до OOM (52ГБ к 3-й эпохе). Один
        # размер -> один граф, память стабильна.
        enc = tok(texts, truncation=True, max_length=MAXLEN, padding="max_length",
                  return_tensors="pt")
        return enc, ys
    return collate


def save_best(model, tok, out):
    """Сохраняем лучший чекпойнт НА ДИСК сразу (device-safe через CPU state_dict):
    если поздняя эпоха упадёт по OOM, лучший уже на диске."""
    import os as _os

    from safetensors.torch import save_file
    _os.makedirs(out, exist_ok=True)
    model.config.save_pretrained(out)
    tok.save_pretrained(out)
    sd = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    # metadata format="pt" обязателен: иначе from_pretrained падает (metadata=None)
    save_file(sd, _os.path.join(out, "model.safetensors"), metadata={"format": "pt"})


@torch.no_grad()
def prob_ai(model, tok, texts, device, bs=64):
    model.eval()
    ps = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt").to(device)
        ps.append(torch.softmax(model(**enc).logits, 1)[:, 1].float().cpu().numpy())
    return np.concatenate(ps) if ps else np.array([])


def val_short_auc(model, tok, val, device):
    h = [r["text"] for r in val if r["label"] == 0]
    a = [r["text"] for r in val if r["label"] == 1]
    aucs = []
    for N in SEL_LENGTHS:
        ph = prob_ai(model, tok, [trunc(t, N) for t in h], device)
        pa = prob_ai(model, tok, [trunc(t, N) for t in a], device)
        aucs.append(roc_auc_score(np.r_[np.zeros(len(ph)), np.ones(len(pa))], np.r_[ph, pa]))
    return float(np.mean(aucs)), [round(x, 4) for x in aucs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bases/ruBert-base")
    ap.add_argument("--out", default=model_path("v3_model"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[device] РЕАЛЬНО считаем на: {device.upper()}  "
          f"(mps.is_available={torch.backends.mps.is_available()}, "
          f"FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')})", flush=True)
    if device != "mps":
        print("[!! ВНИМАНИЕ] НЕ на MPS — обучение будет МЕДЛЕННЫМ. Проверь железо/torch.", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[base] {args.base}  params={nparams/1e6:.0f}M  hidden={model.config.hidden_size}", flush=True)

    tr, val = load("data/v2_train.jsonl"), load("data/v2_val.jsonl")
    items = build_aug(tr)
    print(f"[data] v2_train {len(tr)} -> аугм. {len(items)}  "
          f"labels={dict(Counter(y for _, y in items))}", flush=True)

    dl = DataLoader(DS(items), batch_size=args.bs, shuffle=True, collate_fn=make_collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    lossf = torch.nn.CrossEntropyLoss()

    mps = (device == "mps")
    best_auc, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (enc, y) in enumerate(dl):
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)
            opt.zero_grad()
            loss = lossf(model(**enc).logits, y)
            loss.backward()
            opt.step()
            sch.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"  ep{ep} {step:>4}/{len(dl)} loss {loss.item():.3f}", flush=True)
                if mps:
                    torch.mps.empty_cache()
        dt = time.time() - t0
        if mps:
            torch.mps.empty_cache()
        auc, aucs = val_short_auc(model, tok, val, device)
        print(f"[epoch {ep}] {dt/60:.1f} мин | loss {running/len(dl):.3f} | "
              f"val short-AUC {auc:.4f} (40/30/20={aucs})", flush=True)
        hist.append({"epoch": ep, "minutes": dt / 60, "loss": running / len(dl),
                     "val_short_auc": auc, "val_aucs": aucs})
        if auc > best_auc:
            best_auc = auc
            save_best(model, tok, args.out)          # сразу на диск
            print(f"  -> новый лучший (val short-AUC {auc:.4f}) -> сохранён {args.out}", flush=True)
        if mps:
            torch.mps.empty_cache()

    json.dump({"base": args.base, "best_val_short_auc": best_auc, "history": hist,
               "device": device, "epochs": args.epochs, "bs": args.bs, "lr": args.lr},
              open("data/v3_train_history.json", "w"), ensure_ascii=False, indent=2)
    print(f"[saved] {args.out}  best val short-AUC {best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
