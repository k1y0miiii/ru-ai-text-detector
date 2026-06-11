#!/usr/bin/env python3
"""
train_v4.py — ПЕРЕОБУЧЕНИЕ v3-базы на РАСШИРЕННЫХ данных (v4). Цель — поднять recall
на невиданных генераторах, добавив в train ИИ-тексты Llama 3.1 и Mistral 7B.
gpt4/saiga остаются НЕТРОНУТЫМ hold-out (яблоко-к-яблоку с v3).

Идентично train_v3.py по архитектуре (та же база ai-forever/ruBert-base, та же
аугментация урезанием, фикс-padding MAXLEN, выбор чекпойнта по val short-AUC,
mps.empty_cache после шагов/эпох). Отличие — данные:
  train = data/v2_train.jsonl (= train v3) + data/train_v4_addon.jsonl (новые ИИ)
  val   = data/v2_val.jsonl   (тот же)
Баланс классов: после добавления ИИ доводим human до того же числа дублированием
полных человеческих текстов из train (аугментация урезанием их потом размножит).

Гард по времени (ТЗ): max 90 мин и max 4 эпохи; если эпоха >25 мин — стоп после неё,
лучший чекпойнт уже на диске. v3-артефакты НЕ трогаем (out=v4_model).
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

MAXLEN = 192
AUG_LENGTHS = [75, 50, 40, 30, 20]
SEL_LENGTHS = [40, 30, 20]
MAX_TOTAL_MIN = 90.0          # общий гард по времени
MAX_EPOCH_MIN = 25.0         # если эпоха дольше — стоп после неё


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def wc(t):
    return len(_WORD_RE.findall(t))


def trunc(t, n):
    sp = [(m.start(), m.end()) for m in _WORD_RE.finditer(t)]
    return t.strip() if len(sp) <= n else t[:sp[n - 1][1]].strip()


def build_aug(rows):
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
        enc = tok(texts, truncation=True, max_length=MAXLEN, padding="max_length",
                  return_tensors="pt")
        return enc, ys
    return collate


def save_best(model, tok, out):
    import os as _os

    from safetensors.torch import save_file
    _os.makedirs(out, exist_ok=True)
    model.config.save_pretrained(out)
    tok.save_pretrained(out)
    sd = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
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


def fresh_coat_human(exclude_texts, k):
    """k человеческих текстов coat (label=0, >=200 симв), КОТОРЫХ НЕТ в train.
    Зачем: добавив N новых ИИ, добавляем N новых РЕАЛЬНЫХ human (а не дублей —
    build_aug дедупит дубли) → классы сбалансированы, как в рецепте v3. Источник
    тот же (coat), что и затравки генерации → доменная парность сохранена."""
    if k <= 0:
        return []
    from datasets import load_dataset
    ds = load_dataset("RussianNLP/coat", "binary", split="train")
    import random
    rng = random.Random(2024)
    cand = [t.strip() for t, l in zip(ds["text"], ds["label"])
            if int(l) == 0 and len((t or "").strip()) >= 200 and t.strip() not in exclude_texts]
    rng.shuffle(cand)
    return [{"text": t, "label": 0, "source": "coat", "generator": "coat_human_extra"}
            for t in cand[:k]]


def build_train_rows():
    """v3-train + новые ИИ (llama/mistral) + СТОЛЬКО ЖЕ новых РЕАЛЬНЫХ human из coat
    (баланс классов, как в рецепте v3 — меняем ТОЛЬКО покрытие генераторов)."""
    base = load("data/v2_train.jsonl")
    addon = load("data/train_v4_addon.jsonl")
    n_ai_base = sum(1 for r in base if r["label"] == 1)
    n_hu_base = sum(1 for r in base if r["label"] == 0)
    n_add_ai = len(addon)
    # сколько human добить, чтобы итог был сбалансирован: (hu_base+extra)=(ai_base+add)
    deficit = (n_ai_base + n_add_ai) - n_hu_base
    # исключаем train+val+test (любой класс) — чтобы новые human НЕ протекли в val/test
    exclude = {r["text"].strip() for r in base}
    for p in ("data/v2_val.jsonl", "data/test.jsonl"):
        for r in load(p):
            exclude.add(r["text"].strip())
    extra_hu = fresh_coat_human(exclude, max(0, deficit))
    out = base + addon + extra_hu
    return out, dict(base=len(base), addon_ai=n_add_ai, fresh_human_added=len(extra_hu),
                     n_ai_total=n_ai_base + n_add_ai, n_hu_total=n_hu_base + len(extra_hu))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bases/ruBert-base")
    ap.add_argument("--out", default=model_path("v4_model"))
    ap.add_argument("--epochs", type=int, default=4)
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
        print("[!! ВНИМАНИЕ] НЕ на MPS — обучение будет МЕДЛЕННЫМ.", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[base] {args.base}  params={nparams/1e6:.0f}M  hidden={model.config.hidden_size}", flush=True)

    tr_rows, info = build_train_rows()
    val = load("data/v2_val.jsonl")
    items = build_aug(tr_rows)
    print(f"[data] {info} -> train rows {len(tr_rows)} -> аугм. {len(items)}  "
          f"labels={dict(Counter(y for _, y in items))}", flush=True)

    dl = DataLoader(DS(items), batch_size=args.bs, shuffle=True, collate_fn=make_collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    lossf = torch.nn.CrossEntropyLoss()

    mps = (device == "mps")
    best_auc, hist = -1.0, []
    t_start = time.time()
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
            save_best(model, tok, args.out)
            print(f"  -> новый лучший (val short-AUC {auc:.4f}) -> сохранён {args.out}", flush=True)
        if mps:
            torch.mps.empty_cache()
        # гарды по времени
        elapsed = (time.time() - t_start) / 60
        if dt / 60 > MAX_EPOCH_MIN:
            print(f"[стоп] эпоха {dt/60:.1f} мин > {MAX_EPOCH_MIN} — останавливаюсь, лучший на диске.", flush=True)
            break
        if elapsed > MAX_TOTAL_MIN:
            print(f"[стоп] суммарно {elapsed:.1f} мин > {MAX_TOTAL_MIN} — останавливаюсь.", flush=True)
            break

    json.dump({"base": args.base, "best_val_short_auc": best_auc, "history": hist,
               "device": device, "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
               "data_info": info},
              open("data/v4_train_history.json", "w"), ensure_ascii=False, indent=2)
    print(f"[saved] {args.out}  best val short-AUC {best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
