"""
Генерация машинных текстов через локальную Ollama-модель для генератор-устойчивости.

Зачем: чтобы детектор не был заточен только под RuATD+alpaca, добавляем в машинный
класс тексты от ЕЩЁ ОДНОГО генератора (qwen2.5:3b).

КЛЮЧЕВОЕ — парность тем: промты берём из ЧЕЛОВЕЧЕСКОГО класса coat (label=0), т.е.
генерим машинный абзац на ТУ ЖЕ тему, что у реального человеческого текста. Иначе
классификатор выучит тему/домен, а не авторство (главная ловушка из CLAUDE.md).

Чистка: убираем <think>, преамбулы ("Конечно, вот текст:"), markdown, кавычки.
Разброс почерка: temperature чередуем 0.7 / 1.0.

Запуск:
    python gen_ollama.py --n 300 --out data/qwen3b.jsonl
    python gen_ollama.py --n 4   --out /tmp/sample.jsonl   # быстрый пробник
"""

import argparse
import json
import re
import time
import urllib.request

API = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

SYSTEM = ("Ты пишешь связные тексты в нейтральном энциклопедическом стиле на русском "
          "языке. Никаких рассуждений, вступлений, пояснений, заголовков и markdown — "
          "только готовый абзац.")

PROMPT_TMPL = ("Напиши связный абзац из 4–6 предложений на русском языке на ту же тему, "
               "что и фрагмент ниже. Выведи ТОЛЬКО текст абзаца.\n\nФрагмент: {seed}")

# преамбулы, которые иногда лезут в начало (ведущая мета-фраза до двоеточия).
# {0,90} — чтобы ловить длинные вроде "Вот связный абзац на основе фрагмента:".
PREAMBLE = re.compile(
    r"^\s*(конечно|разумеется|хорошо|вот|ниже|текст абзаца|абзац)\b[^:\n]{0,90}:\s*",
    re.IGNORECASE)
REFUSAL = re.compile(r"^\s*(извин|я не мог|как языков|не могу)", re.IGNORECASE)


def clean(s: str) -> str:
    # think-блоки (страховка; qwen2.5:3b их не использует)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    s = re.sub(r"</?think>", "", s, flags=re.I)
    # markdown-мусор
    s = re.sub(r"[#*`>_]+", "", s)
    s = s.strip()
    # преамбулы (до двух раз — иногда вложенные)
    for _ in range(2):
        s2 = PREAMBLE.sub("", s)
        if s2 == s:
            break
        s = s2.strip()
    # обрамляющие кавычки
    s = s.strip().strip('"«»“”').strip()
    # схлопнуть пробелы
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cyrillic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return cyr / len(letters)


def valid(s: str) -> bool:
    if len(s) < 200:                      # v1 — только длинные тексты
        return False
    if "<think>" in s.lower() or "</think>" in s.lower():
        return False
    if REFUSAL.match(s):
        return False
    if cyrillic_ratio(s) < 0.5:           # отсекаем англоязычные/смешанные выводы
        return False
    return True


def generate(seed: str, temperature: float) -> str:
    body = json.dumps({
        "model": MODEL,
        "system": SYSTEM,
        "prompt": PROMPT_TMPL.format(seed=seed[:300]),
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 300, "top_p": 0.9},
    }).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"]


def load_seeds():
    """Темы = человеческие тексты coat (label=0, >=200 симв) — для парности."""
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
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/qwen3b.jsonl")
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(123)
    seeds = load_seeds()
    random.shuffle(seeds)
    seeds = seeds[args.seed_offset:]
    print(f"тем-затравок (человеческих, coat): {len(seeds)}")

    out = []
    attempts = 0
    t0 = time.time()
    si = 0
    while len(out) < args.n and si < len(seeds) and attempts < args.n * 3:
        seed = seeds[si]
        si += 1
        attempts += 1
        temp = 0.7 if (len(out) % 2 == 0) else 1.0   # чередуем почерк
        try:
            raw = generate(seed, temp)
        except Exception as e:
            print(f"[skip] gen error: {e}")
            continue
        txt = clean(raw)
        if valid(txt):
            out.append({"text": txt, "label": 1, "source": "qwen3b",
                        "generator": "qwen3b", "temp": temp})
            if len(out) % 25 == 0:
                dt = time.time() - t0
                print(f"  {len(out)}/{args.n}  ({dt/len(out):.1f}s/текст, "
                      f"ещё ~{(args.n-len(out))*dt/len(out)/60:.1f} мин)")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"\nГотово: {len(out)} текстов за {dt/60:.1f} мин -> {args.out} "
          f"(попыток {attempts})")


if __name__ == "__main__":
    main()
