#!/usr/bin/env python3
"""
build_data_v2.py — данные для трансформера v2.

Микс тот же, что у feature-based (coat + alpaca + qwen3b, человек из coat), НО с
добавлением КОРОТКИХ (<200 симв) текстов из coat обоих классов — трансформер
читает токены напрямую и на коротких не вырождается (там был слепой угол v1).

Честность сравнения и генерализации:
  * длинную часть (train/val/test, >=200) берём РОВНО из v1 (apples-to-apples);
  * GPT-4 / saiga (data/holdout_*.jsonl) в train НЕ кладём — это критерий №1 (recall
    на невиданном генераторе), мерим отдельно;
  * короткие добавляем из coat (оба класса, парные темы — мало доменной утечки),
    дизъюнктно по тексту с длинной частью и между train/val/test.
"""

import hashlib
import json
import random
from collections import Counter

SEED = 42
rng = random.Random(SEED)
SHORT_MIN, SHORT_MAX = 30, 200       # «короткий» = 30..199 симв
N_SHORT = {"train": 400, "val": 75, "test": 150}   # на КЛАСС


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def md5(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def main():
    long_tr = load("data/train.jsonl")
    long_va = load("data/val.jsonl")
    long_te = load("data/test.jsonl")
    seen = {md5(r["text"]) for r in long_tr + long_va + long_te}

    # короткие из coat
    from datasets import load_dataset
    ds = load_dataset("RussianNLP/coat", "binary", split="train")
    short = {0: [], 1: []}
    for t, l in zip(ds["text"], ds["label"]):
        t = (t or "").strip()
        if SHORT_MIN <= len(t) < SHORT_MAX:
            h = md5(t)
            if h not in seen:
                seen.add(h)
                short[int(l)].append({"text": t, "label": int(l),
                                      "source": "coat", "generator":
                                      "human" if int(l) == 0 else "coat_machine"})
    rng.shuffle(short[0]); rng.shuffle(short[1])

    # нарезаем короткие на train/val/test (поровну по классам)
    cut = {}
    i0 = i1 = 0
    for part in ("train", "val", "test"):
        n = N_SHORT[part]
        cut[part] = short[0][i0:i0 + n] + short[1][i1:i1 + n]
        i0 += n; i1 += n
        rng.shuffle(cut[part])

    v2_train = long_tr + cut["train"]
    v2_val = long_va + cut["val"]
    rng.shuffle(v2_train); rng.shuffle(v2_val)

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok] {path}: {len(rows)} (human {sum(r['label']==0 for r in rows)}, "
              f"ai {sum(r['label']==1 for r in rows)})")

    dump("data/v2_train.jsonl", v2_train)
    dump("data/v2_val.jsonl", v2_val)
    dump("data/short_test.jsonl", cut["test"])

    import statistics
    L = [len(r["text"]) for r in v2_train]
    print(f"v2_train длины: min={min(L)} median={int(statistics.median(L))} max={max(L)}")
    print("v2_train AI по generator:",
          dict(Counter(r.get("generator") for r in v2_train if r["label"] == 1)))
    print(f"short_test: {len(cut['test'])} (по {N_SHORT['test']} на класс), "
          "GPT-4/saiga в train НЕ входят (held-out отдельно)")


if __name__ == "__main__":
    main()
