"""
Подготовка обучающего датасета для детектора ИИ-текста.

Что делает:
  1. Загружает выбранные датасеты (HF + локальные human-тексты до 2022)
  2. Приводит всё к единому формату {"text", "label", "source", "generator"}
  3. Фильтрует по длине (короткие тексты выкидываем — на них признаки шумят)
  4. Дедуплицирует
  5. Балансирует классы (одинаковое число human/ai)
  6. Делит на train/val/test со стратификацией
  7. Проверяет, не разъехались ли распределения длин между классами
     (если разъехались — модель выучит длину/тему, а не авторство — варнинг)

Запуск:
    python prepare_data.py
    python prepare_data.py --max-per-class 5000 --min-chars 200

ВАЖНО про источники:
  - HF-датасеты могут менять формат полей. Адаптеры ниже (LOADERS) — это место,
    которое надо проверить под конкретную версию датасета. Если поле называется
    иначе — поправь маппинг в соответствующем адаптере.
  - Человеческие локальные тексты клади в data/human_raw/*.txt или *.jsonl
    (всё, что гарантированно написано до конца 2022).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

random.seed(42)

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR
HUMAN_RAW = DATA_DIR / "human_raw"   # сюда складывай локальные человеческие тексты


# ---------------------------------------------------------------------------
# Адаптеры под конкретные датасеты.
# Каждый возвращает список dict: {"text", "label", "source", "generator"}
#   label: 0 = человек, 1 = ИИ
#   generator: какая модель сгенерировала (для ИИ) или "human"
# Включай нужные через --sources.
# ---------------------------------------------------------------------------

def _row(text, label, source, generator):
    return {"text": text, "label": label, "source": source, "generator": generator}


def load_ai_text_pile(limit: int | None):
    """artem9k/ai-text-detection-pile — эссе/long-form, EN. Поля: text, source(human/ai)."""
    from datasets import load_dataset
    ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
    out, n = [], 0
    for r in ds:
        text = (r.get("text") or "").strip()
        src = (r.get("source") or "").lower()
        if not text:
            continue
        label = 0 if "human" in src else 1
        out.append(_row(text, label, "ai-text-detection-pile", "human" if label == 0 else "mixed_llm"))
        n += 1
        if limit and n >= limit:
            break
    return out


def load_defactify(limit: int | None):
    """Rajarshi-Roy-research/Defactify_Text_Dataset — NYT + 6 LLM. Проверь имена полей!"""
    from datasets import load_dataset
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Text_Dataset", split="train", streaming=True)
    out, n = [], 0
    for r in ds:
        # Имена полей в этом датасете могут отличаться — самое частое: human_text / <model>_text.
        # Берём универсально: всё, что заканчивается на _text, кроме human — это ИИ.
        for key, val in r.items():
            if not isinstance(val, str) or not val.strip() or not key.endswith("text"):
                continue
            label = 0 if "human" in key.lower() else 1
            gen = "human" if label == 0 else key.replace("_text", "")
            out.append(_row(val.strip(), label, "defactify", gen))
            n += 1
            if limit and n >= limit:
                return out
    return out


def load_ru_turbo_alpaca(limit: int | None):
    """IlyaGusev/ru_turbo_alpaca — ChatGPT-генерация на русском => класс ИИ=1.

    Тонкость: датасет публикуется через dataset-скрипт (ru_turbo_alpaca.py), а
    datasets>=4.0 скрипты больше НЕ поддерживает ("Dataset scripts are no longer
    supported"). Поэтому грузим авто-сконвертированную parquet-ветку
    refs/convert/parquet — это ровно те же данные, но без скрипта.

    Поля (проверено через load_dataset(...).take(1)):
      instruction, input, output (ответ ChatGPT), alternative_output, label(качество).
    Берём output как основной ИИ-текст, instruction — запасной вариант.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("IlyaGusev/ru_turbo_alpaca",
                          revision="refs/convert/parquet", split="train")
    except Exception:
        # запасной путь для старых datasets (<4.0), где ещё работает скрипт
        ds = load_dataset("IlyaGusev/ru_turbo_alpaca", split="train",
                          trust_remote_code=True)
    out = []
    for r in ds:
        text = (r.get("output") or r.get("instruction") or "").strip()
        if text:
            out.append(_row(text, 1, "ru_turbo_alpaca", "chatgpt"))
        if limit and len(out) >= limit:
            break
    return out


def load_local_human(limit: int | None):
    """Локальные гарантированно человеческие тексты (до 2022) из data/human_raw/."""
    out = []
    if not HUMAN_RAW.exists():
        print(f"[i] {HUMAN_RAW} не найдена — пропускаю локальные человеческие тексты")
        return out
    for p in sorted(HUMAN_RAW.glob("*")):
        if p.suffix == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    t = (obj.get("text") or "").strip()
                    if t:
                        out.append(_row(t, 0, f"local:{p.name}", "human"))
        elif p.suffix == ".txt":
            t = p.read_text(encoding="utf-8").strip()
            if t:
                out.append(_row(t, 0, f"local:{p.name}", "human"))
        if limit and len(out) >= limit:
            break
    return out


def load_coat(limit: int | None):
    """RuATD / CoAT (RussianNLP/coat), конфиг binary — детекция искусственного
    текста на русском. ЛУЧШИЙ источник для нашей задачи: содержит ОБА класса
    (человек + машина) на сопоставимых темах (человеческая часть — Wikipedia/
    новости/литература, машинная — парафразы/переводы/генерация на тех же текстах),
    поэтому почти нет доменной утечки, в отличие от пары "новости vs ответы ChatGPT".

    Поля (проверено): id, text, label. label КАК ЕСТЬ: 0=человек, 1=машина —
    подтверждено на конкретных строках (label=0 — связный человеческий текст,
    label=1 — машинный парафраз/перевод).

    Streaming на этом датасете спотыкается (Parquet._generate_tables кидает
    исключение и подвисает), поэтому грузим ОБЫЧНЫМ load_dataset (binary ~75 МБ,
    кешируется один раз).

    limit здесь намеренно НЕ ограничиваем: фильтр по длине в main() отрезает ~2/3
    коротких текстов RuATD, поэтому отдаём весь train-пул, а финальный размер
    каждого класса задаёт --max-per-class через balance().
    """
    from datasets import load_dataset
    ds = load_dataset("RussianNLP/coat", "binary", split="train")  # БЕЗ streaming
    cols = ds.column_names
    # в binary-конфиге поля генератора нет; если в будущем подключат authorship —
    # подхватим его автоматически
    has_auth = "authorship" in cols
    texts = ds["text"]
    labels = ds["label"]
    auths = ds["authorship"] if has_auth else None
    out = []
    for i in range(len(texts)):
        text = (texts[i] or "").strip()
        if not text:
            continue
        label = int(labels[i])                       # 0=человек, 1=машина — как есть
        if has_auth and auths[i]:
            gen = auths[i]
        else:
            gen = "human" if label == 0 else "coat_machine"
        out.append(_row(text, label, "coat", gen))
    return out


def load_qwen3b(limit: int | None):
    """Локально сгенерированные машинные тексты (qwen2.5:3b) из data/qwen3b.jsonl.
    Создаётся скриптом gen_ollama.py (темы парные к человеческому классу coat).
    Нужен для генератор-устойчивости: машинный класс должен покрывать не только
    RuATD-парафразы и ChatGPT-alpaca, но и свежий локальный генератор."""
    path = DATA_DIR / "qwen3b.jsonl"
    if not path.exists():
        print(f"[i] {path} не найден — сначала запусти gen_ollama.py")
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        t = (obj.get("text") or "").strip()
        if t:
            out.append(_row(t, 1, "qwen3b", obj.get("generator", "qwen3b")))
        if limit and len(out) >= limit:
            break
    return out


LOADERS = {
    "pile": load_ai_text_pile,
    "defactify": load_defactify,
    "ru_alpaca": load_ru_turbo_alpaca,
    "local_human": load_local_human,
    "coat": load_coat,
    "qwen3b": load_qwen3b,
}


# ---------------------------------------------------------------------------
# Обработка
# ---------------------------------------------------------------------------

def dedup(rows):
    seen, out = set(), []
    for r in rows:
        h = hashlib.md5(r["text"].encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out


def filter_len(rows, min_chars):
    return [r for r in rows if len(r["text"]) >= min_chars]


def cap_machine_per_source(rows, cap):
    """Ограничить число МАШИННЫХ примеров (label=1) на каждый source.

    Нужно для контролируемого микса: машинный класс должен покрывать И парафразы/
    переводы (coat), И штампованный ChatGPT (alpaca). Без кэпа balance() насыпал бы
    ИИ-класс почти целиком из coat (его ~32k против ~1.7k у alpaca), и ChatGPT-штамп
    в обучение практически не попал бы. Человеческий класс (label=0) не трогаем.
    """
    from collections import defaultdict
    counts = defaultdict(int)
    random.shuffle(rows)   # кэп берёт СЛУЧАЙНОЕ подмножество, а не первые N
    out = []
    for r in rows:
        if r["label"] == 1:
            if counts[r["source"]] >= cap:
                continue
            counts[r["source"]] += 1
        out.append(r)
    return out


def balance(rows, max_per_class):
    human = [r for r in rows if r["label"] == 0]
    ai = [r for r in rows if r["label"] == 1]
    random.shuffle(human)
    random.shuffle(ai)
    k = min(len(human), len(ai))
    if max_per_class:
        k = min(k, max_per_class)
    if k == 0:
        raise SystemExit("Один из классов пуст. Нужны и человеческие, и ИИ тексты.")
    return human[:k] + ai[:k]


def warn_if_length_leak(rows):
    """Если средние длины классов сильно расходятся — детектор выучит длину."""
    hl = [len(r["text"]) for r in rows if r["label"] == 0]
    al = [len(r["text"]) for r in rows if r["label"] == 1]
    if not hl or not al:
        return
    mh, ma = statistics.mean(hl), statistics.mean(al)
    ratio = max(mh, ma) / max(1, min(mh, ma))
    print(f"[len] human≈{mh:.0f} симв, ai≈{ma:.0f} симв, ratio={ratio:.2f}")
    if ratio > 1.5:
        print("[!] ВНИМАНИЕ: длины классов сильно различаются. "
              "Модель может выучить длину вместо авторства. "
              "Подрежь тексты до сопоставимой длины или подбери парные по теме.")


def split(rows, val=0.1, test=0.1):
    random.shuffle(rows)
    n = len(rows)
    n_test = int(n * test)
    n_val = int(n * val)
    return rows[n_test + n_val:], rows[n_test:n_test + n_val], rows[:n_test]


def write_jsonl(rows, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] {path}  ({len(rows)} примеров)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["local_human", "ru_alpaca"],
                    choices=list(LOADERS.keys()),
                    help="какие источники грузить")
    ap.add_argument("--max-per-class", type=int, default=5000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--machine-cap-per-source", type=int, default=None,
                    help="макс. МАШИННЫХ примеров (label=1) с каждого source — "
                         "для контролируемого микса (напр. 700 coat + 700 alpaca)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    for s in args.sources:
        print(f"[load] {s} ...")
        try:
            got = LOADERS[s](args.max_per_class * 2)
            print(f"       +{len(got)}")
            rows += got
        except Exception as e:
            print(f"[!] источник {s} не загрузился: {e}")

    print(f"\nвсего сырых: {len(rows)}")
    rows = filter_len(dedup(rows), args.min_chars)
    print(f"после dedup+filter: {len(rows)}")

    if args.machine_cap_per_source:
        rows = cap_machine_per_source(rows, args.machine_cap_per_source)
        from collections import Counter
        comp = Counter(r["source"] for r in rows if r["label"] == 1)
        print(f"состав ИИ-класса после кэпа по источникам: {dict(comp)}")

    rows = balance(rows, args.max_per_class)
    print(f"после balance: {len(rows)} "
          f"(human={sum(1 for r in rows if r['label']==0)}, "
          f"ai={sum(1 for r in rows if r['label']==1)})")

    warn_if_length_leak(rows)

    tr, va, te = split(rows)
    write_jsonl(tr, OUT_DIR / "train.jsonl")
    write_jsonl(va, OUT_DIR / "val.jsonl")
    write_jsonl(te, OUT_DIR / "test.jsonl")
    print("\nГотово. Дальше: python train.py data/train.jsonl")


if __name__ == "__main__":
    main()
