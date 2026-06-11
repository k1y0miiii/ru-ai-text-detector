#!/usr/bin/env python3
"""
build_data_v1_1.py — данные для эксперимента v1.1: добавляем в МАШИННЫЙ класс
генераторы, которых детектор не видел (GPT-4 из ru_instruct_gpt4, gpt-3.5 из
ru_turbo_saiga), сохраняя баланс и размер train как у v1. features/архитектуру
не трогаем — это эксперимент с ДАННЫМИ.

Честность генерализации (ключевое):
  * 90 GPT-4 + 90 saiga из data/robustness_raw.jsonl (bucket B) — HOLD-OUT, в train
    НЕ попадают (на них меряем recall на «невиданном» тексте без утечки; они же —
    батч measure_robustness, поэтому его перезапуск тоже без утечки);
  * в train берём СВЕЖИЕ GPT-4/saiga, дизъюнктные с этим hold-out;
  * val.jsonl / test.jsonl НЕ трогаем (свои генераторы) — сравнение с v1 «яблоко-в-яблоко».

Баланс: human не меняем (1038), AI оставляем = human, ДОБАВИВ gpt4+saiga и
ПРОПОРЦИОНАЛЬНО проредив исходные генераторы (чтобы общий размер/баланс как у v1).
"""

import io
import json
import random
from collections import Counter, defaultdict

import zstandard
from huggingface_hub import hf_hub_download

SEED = 42
GPT4_TRAIN = 200
SAIGA_TRAIN = 200
MIN_CHARS, MAX_CHARS = 200, 3000
rng = random.Random(SEED)


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def gpt4_texts():
    p = hf_hub_download("lksy/ru_instruct_gpt4", "ru_instruct_gpt4.jsonl", repo_type="dataset")
    out, seen = [], set()
    for line in open(p, encoding="utf-8"):
        t = (json.loads(line).get("output") or "").strip()
        if MIN_CHARS <= len(t) <= MAX_CHARS and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def saiga_texts():
    p = hf_hub_download("IlyaGusev/ru_turbo_saiga", "ru_turbo_saiga.jsonl.zst", repo_type="dataset")
    dctx = zstandard.ZstdDecompressor()
    out, seen = [], set()
    with open(p, "rb") as fh:
        rd = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8")
        for line in rd:
            for m in json.loads(line).get("messages", []):
                if m.get("role") == "bot":
                    t = (m.get("content") or "").strip()
                    if MIN_CHARS <= len(t) <= MAX_CHARS and t not in seen:
                        seen.add(t)
                        out.append(t)
            if len(out) >= 3000:
                break
    return out


def main():
    # --- hold-out из robustness-батча (НЕ в train) ---
    rob = load_jsonl("data/robustness_raw.jsonl")
    hold_gpt4 = [r["text"] for r in rob if r["source"] == "ru_instruct_gpt4"]
    hold_saiga = [r["text"] for r in rob if r["source"] == "ru_turbo_saiga"]
    held = set(hold_gpt4) | set(hold_saiga)

    def dump(path, texts, gen):
        with open(path, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t, "label": 1, "generator": gen}, ensure_ascii=False) + "\n")

    dump("data/holdout_gpt4.jsonl", hold_gpt4, "gpt4_instruct")
    dump("data/holdout_saiga.jsonl", hold_saiga, "chatgpt_saiga")

    # --- свежие train-порции GPT-4 / saiga, дизъюнктные с hold-out ---
    gpt4_fresh = [t for t in gpt4_texts() if t not in held][:GPT4_TRAIN]
    saiga_fresh = [t for t in saiga_texts() if t not in held][:SAIGA_TRAIN]

    # --- исходный train v1: humans + AI по генераторам ---
    tr = load_jsonl("data/train.jsonl")
    humans = [r for r in tr if r["label"] == 0]
    ai_by = defaultdict(list)
    for r in tr:
        if r["label"] == 1:
            ai_by[r.get("generator", "?")].append(r)

    n_human = len(humans)
    n_added = len(gpt4_fresh) + len(saiga_fresh)
    keep_orig = n_human - n_added                       # сколько исходного AI оставить
    orig_total = sum(len(v) for v in ai_by.values())

    # пропорциональное прореживание исходных генераторов до keep_orig
    kept = []
    for g, rows in ai_by.items():
        k = round(keep_orig * len(rows) / orig_total)
        kept += rng.sample(rows, min(k, len(rows)))
    # подгон до точного keep_orig
    rng.shuffle(kept)
    kept = kept[:keep_orig]

    new_ai = (kept
              + [{"text": t, "label": 1, "generator": "gpt4_instruct"} for t in gpt4_fresh]
              + [{"text": t, "label": 1, "generator": "chatgpt_saiga"} for t in saiga_fresh])
    train = humans + new_ai
    rng.shuffle(train)

    with open("data/train_v1_1.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # отчёт
    import statistics
    ai = [r for r in train if r["label"] == 1]
    print(f"train_v1_1: всего {len(train)}  human {len(humans)}  AI {len(ai)}")
    print("AI по generator:", dict(Counter(r.get("generator") for r in ai)))
    L = [len(r["text"]) for r in train]
    print(f"len train: min={min(L)} median={int(statistics.median(L))} max={max(L)}")
    print(f"hold-out: gpt4 {len(hold_gpt4)} (медиана len {int(statistics.median([len(t) for t in hold_gpt4]))}), "
          f"saiga {len(hold_saiga)} (медиана len {int(statistics.median([len(t) for t in hold_saiga]))})")
    print("проверка дизъюнктности train∩holdout:",
          len(set(r['text'] for r in ai) & held), "(должно быть 0)")


if __name__ == "__main__":
    main()
