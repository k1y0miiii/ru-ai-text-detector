#!/usr/bin/env python3
"""
build_robustness_batch.py — собирает батч ИИ-текстов для замера устойчивости
детектора (measure_robustness.py). Только сборка данных, детектор тут не участвует.

Источники:
  A (test.jsonl, label=1) — ИИ-тексты, которые детектор НЕ видел при обучении, но
    из ТЕХ ЖЕ генераторов, что в обучении (chatgpt-alpaca / qwen3b / coat_machine).
  B (HuggingFace) — «чужой» ИИ:
    - IlyaGusev/ru_turbo_saiga (role=bot, model gpt-3.5-turbo) — ChatGPT, но другой
      датасет/промпты, чем alpaca в обучении;
    - lksy/ru_instruct_gpt4 (поле output) — GPT-4, генератор, КОТОРОГО в обучении
      НЕ было вовсе (самый честный «невиданный» случай).

Берём первые N подходящих (порядок файла) — детерминированно, без перемешивания.
Поле bucket: "A" / "B" — чтобы потом сравнить recall на «своём» vs «чужом» ИИ.
"""

import io
import json
from pathlib import Path

import zstandard
from huggingface_hub import hf_hub_download

MIN_CHARS = 200
PER_B_SOURCE = 90          # ~80-100 из каждого датасета B
OUT = Path("data/robustness_raw.jsonl")


def from_test():
    rows = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
    out = []
    for r in rows:
        if r.get("label") == 1 and len(r["text"]) >= MIN_CHARS:
            out.append({"text": r["text"], "generator": r.get("generator", "?"),
                        "bucket": "A", "source": "test.jsonl/" + r.get("source", "?")})
    return out


def from_saiga(limit):
    p = hf_hub_download("IlyaGusev/ru_turbo_saiga", "ru_turbo_saiga.jsonl.zst",
                        repo_type="dataset")
    dctx = zstandard.ZstdDecompressor()
    out, seen = [], set()
    with open(p, "rb") as fh:
        reader = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8")
        for line in reader:
            if len(out) >= limit:
                break
            rec = json.loads(line)
            for m in rec.get("messages", []):
                if m.get("role") == "bot":
                    t = (m.get("content") or "").strip()
                    if len(t) >= MIN_CHARS and t not in seen:
                        seen.add(t)
                        out.append({"text": t, "generator": "chatgpt_saiga",
                                    "bucket": "B", "source": "ru_turbo_saiga"})
                        if len(out) >= limit:
                            break
    return out


def from_gpt4(limit):
    p = hf_hub_download("lksy/ru_instruct_gpt4", "ru_instruct_gpt4.jsonl",
                        repo_type="dataset")
    out, seen = [], set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            if len(out) >= limit:
                break
            rec = json.loads(line)
            t = (rec.get("output") or "").strip()   # ответ модели (GPT-4)
            if len(t) >= MIN_CHARS and t not in seen:
                seen.add(t)
                out.append({"text": t, "generator": "gpt4_instruct",
                            "bucket": "B", "source": "ru_instruct_gpt4"})
    return out


def main():
    batch = from_test() + from_saiga(PER_B_SOURCE) + from_gpt4(PER_B_SOURCE)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in batch:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    import statistics
    print(f"всего: {len(batch)} -> {OUT}")
    print("по bucket:    ", dict(Counter(r["bucket"] for r in batch)))
    print("по generator: ", dict(Counter(r["generator"] for r in batch)))
    print("по source:    ", dict(Counter(r["source"] for r in batch)))
    L = [len(r["text"]) for r in batch]
    print(f"длины: min={min(L)} median={int(statistics.median(L))} max={max(L)}")


if __name__ == "__main__":
    main()
