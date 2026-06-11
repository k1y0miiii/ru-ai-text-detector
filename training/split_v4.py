#!/usr/bin/env python3
"""
split_v4.py — фаза 1.3: разбиваем сырые llama/mistral-генерации на train-addon и
hold-out ПО КАЖДОМУ генератору отдельно (70/30), дизъюнктно по тексту.

  data/llama3_raw.jsonl   -> 70% в train-addon, 30% -> data/holdout_llama.jsonl
  data/mistral_raw.jsonl  -> 70% в train-addon, 30% -> data/holdout_mistral.jsonl
  train-addon обоих генераторов -> data/train_v4_addon.jsonl

Дедуп по точному тексту ВНУТРИ генератора (на всякий случай). Проверяем дизъюнктность
train/holdout по тексту. Детерминированный shuffle (seed=2024).
"""
import json
import random


def load(p):
    try:
        return [json.loads(l) for l in open(p, encoding="utf-8")]
    except FileNotFoundError:
        return []


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        t = r["text"].strip()
        if t and t not in seen:
            seen.add(t)
            out.append(r)
    return out


def main():
    rng = random.Random(2024)
    addon = []
    summary = {}
    for gen, raw_path, hold_path in [
        ("llama3", "data/llama3_raw.jsonl", "data/holdout_llama.jsonl"),
        ("mistral", "data/mistral_raw.jsonl", "data/holdout_mistral.jsonl"),
    ]:
        rows = dedup(load(raw_path))
        rng.shuffle(rows)
        ntr = int(round(len(rows) * 0.70))
        tr, ho = rows[:ntr], rows[ntr:]
        # дизъюнктность по тексту
        assert not (set(r["text"] for r in tr) & set(r["text"] for r in ho)), \
            f"{gen}: train/holdout пересеклись по тексту!"
        with open(hold_path, "w", encoding="utf-8") as f:
            for r in ho:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        addon.extend(tr)
        summary[gen] = dict(raw=len(rows), train=len(tr), holdout=len(ho))
        print(f"[{gen}] raw(dedup)={len(rows)} -> train={len(tr)}  holdout={len(ho)} -> {hold_path}")

    rng.shuffle(addon)
    with open("data/train_v4_addon.jsonl", "w", encoding="utf-8") as f:
        for r in addon:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[train-addon] {len(addon)} ИИ-текстов -> data/train_v4_addon.jsonl")
    print(f"[summary] {json.dumps(summary, ensure_ascii=False)}")
    total_new = sum(s['raw'] for s in summary.values())
    print(f"[total new AI texts] {total_new}")
    return total_new


if __name__ == "__main__":
    main()
