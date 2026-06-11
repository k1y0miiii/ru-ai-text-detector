"""
gen_ollama_v4.py — генерация машинных текстов через произвольную локальную Ollama-модель
для РАСШИРЕНИЯ покрытия генераторов (v4). По мотивам gen_ollama.py, но:
  - модель параметризуется (--model llama3.1:8b / mistral:7b);
  - тексты ДЛИННЕЕ (целимся в 200-600 слов: развёрнутый текст из нескольких абзацев),
    т.к. v3/v4-аугментация урезанием сама делает короткие версии — а длинная база нужна;
  - жёсткий wall-clock лимит (--minutes), инкрементальная запись (если убьют — не потеряем);
  - temperature чередуем 0.7/0.8/0.9, top_p 0.9.

КЛЮЧЕВОЕ (как в gen_ollama.py): парность тем — затравки берём из ЧЕЛОВЕЧЕСКОГО coat
(label=0, >=200 симв). Иначе классификатор выучит тему, а не авторство.

Запуск:
    python gen_ollama_v4.py --model llama3.1:8b --out data/llama3_raw.jsonl --n 400 --minutes 35
"""

import argparse
import json
import os
import re
import time
import urllib.request

API = "http://localhost:11434/api/generate"

SYSTEM = ("Ты пишешь связные, развёрнутые тексты в нейтральном энциклопедическом стиле "
          "на русском языке. Никаких рассуждений, вступлений, пояснений, заголовков и "
          "markdown — только готовый текст из нескольких абзацев.")

PROMPT_TMPL = ("Напиши развёрнутый связный текст из 3–5 абзацев (примерно 250–500 слов) "
               "на русском языке на ту же тему, что и фрагмент ниже. Раскрой тему "
               "содержательно. Выведи ТОЛЬКО текст, без заголовков и пояснений.\n\n"
               "Фрагмент: {seed}")

PREAMBLE = re.compile(
    r"^\s*(конечно|разумеется|хорошо|вот|ниже|текст|абзац)\b[^:\n]{0,90}:\s*",
    re.IGNORECASE)
REFUSAL = re.compile(r"^\s*(извин|я не мог|как языков|не могу|i\b|sorry|as an)", re.IGNORECASE)
WR = re.compile(r"\w+", re.U)


def clean(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    s = re.sub(r"</?think>", "", s, flags=re.I)
    s = re.sub(r"[#*`>_]+", "", s)          # markdown-мусор
    s = s.strip()
    for _ in range(2):                       # преамбулы (иногда вложенные)
        s2 = PREAMBLE.sub("", s)
        if s2 == s:
            break
        s = s2.strip()
    s = s.strip().strip('"«»“”').strip()
    # схлопываем только горизонтальные пробелы, абзацы (\n\n) сохраняем как один пробел
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s).strip()
    return s


def cyrillic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return cyr / len(letters)


def valid(s: str) -> bool:
    # целимся в длинные; принимаем >=130 слов, чтобы не выбрасывать пограничное
    if len(WR.findall(s)) < 130:
        return False
    if "<think>" in s.lower() or "</think>" in s.lower():
        return False
    if REFUSAL.match(s):
        return False
    if cyrillic_ratio(s) < 0.6:
        return False
    return True


def generate(model: str, seed: str, temperature: float) -> str:
    body = json.dumps({
        "model": model,
        "system": SYSTEM,
        "prompt": PROMPT_TMPL.format(seed=seed[:300]),
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 750, "top_p": 0.9},
    }).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["response"]


def load_seeds():
    from datasets import load_dataset
    ds = load_dataset("RussianNLP/coat", "binary", split="train")
    seeds = []
    for t, l in zip(ds["text"], ds["label"]):
        t = (t or "").strip()
        if int(l) == 0 and len(t) >= 200:
            seeds.append(t)
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--minutes", type=float, default=35.0)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    gen_name = args.model.split(":")[0]          # llama3.1 / mistral
    import random
    random.seed(123)
    seeds = load_seeds()
    random.shuffle(seeds)
    seeds = seeds[args.seed_offset:]
    print(f"[{args.model}] тем-затравок (человеческих, coat): {len(seeds)}", flush=True)

    temps = [0.7, 0.8, 0.9]
    out, attempts, si = [], 0, 0
    t0 = time.time()
    deadline = t0 + args.minutes * 60

    def flush_out():
        with open(args.out, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    while len(out) < args.n and si < len(seeds) and time.time() < deadline:
        seed = seeds[si]
        si += 1
        attempts += 1
        temp = temps[len(out) % len(temps)]
        try:
            raw = generate(args.model, seed, temp)
        except Exception as e:
            print(f"[skip] gen error: {e}", flush=True)
            continue
        txt = clean(raw)
        if valid(txt):
            out.append({"text": txt, "label": 1, "source": gen_name,
                        "generator": gen_name, "temp": temp})
            if len(out) % 10 == 0:
                flush_out()                      # инкрементальный сейв
                dt = time.time() - t0
                left_min = (deadline - time.time()) / 60
                print(f"  [{args.model}] {len(out)}/{args.n}  "
                      f"({dt/len(out):.1f}s/текст, осталось {left_min:.1f} мин wall)", flush=True)

    flush_out()
    dt = time.time() - t0
    reason = "n" if len(out) >= args.n else ("seeds" if si >= len(seeds) else "time")
    print(f"\n[{args.model}] ГОТОВО: {len(out)} текстов за {dt/60:.1f} мин -> {args.out} "
          f"(попыток {attempts}, стоп по: {reason})", flush=True)


if __name__ == "__main__":
    main()
