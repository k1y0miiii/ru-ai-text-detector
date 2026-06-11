#!/usr/bin/env python3
"""
detect_cli.py — командный интерфейс к детектору ИИ-текста.

Боевая модель — v3 (дообученный ruBert-base). Доступны также v1 (признаки, быстрый,
интерпретируемый) и v2 (tiny2, лёгкий откат) через --model.

Режимы:
  python detect_cli.py диплом.txt                  # один файл (модель v3)
  python detect_cli.py --text "..." --model v1     # текст из аргумента, модель v1
  python detect_cli.py --dir samples/              # батч по папке (+ скоринг по labels.json)

Тот же конвейер, что у сервиса app.py: v1 = features+model.joblib; v2/v3 = transformer_score.
Порог калиброван под FPR<=3%; на коротких фрагментах (<40 слов) — строже (length-aware).
Признаки v1 показываем всегда (прозрачность), даже при вердикте v3.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib

from aidetector import transformer_score as TF
from aidetector.features import FEATURE_NAMES, extract, to_vector
from aidetector.paths import model_path

V1_MODEL_PATH = model_path("model.joblib")
V1_THRESHOLD = 0.86
MIN_RELIABLE_CHARS = 200


class C:
    R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GRN = "\033[32m"; YEL = "\033[33m"; CYAN = "\033[36m"


def col(s, c):
    return f"{c}{s}{C.R}"


_V1 = None


def v1_model():
    global _V1
    if _V1 is None:
        try:
            _V1 = joblib.load(V1_MODEL_PATH)
        except FileNotFoundError:
            sys.exit(col(f"[x] Не найден {V1_MODEL_PATH}. Обучи v1 (finalize_v1.py).", C.RED))
    return _V1


def score(model, text):
    """-> (proba, threshold, vec, short_chars). vec — фичи v1 (для показа всегда)."""
    vec = to_vector(extract(text))
    nwords = TF.word_count(text)
    if model == "v1":
        proba = float(v1_model().predict_proba([vec])[0][1])
        thr = V1_THRESHOLD
    else:
        sc = TF.get(model)
        proba = float(sc.proba([text])[0])
        thr = sc.threshold_for(nwords)
    return proba, thr, vec, len(text) < MIN_RELIABLE_CHARS


def prob_bar(proba, thr, width=32):
    fill = int(round(proba * width))
    tpos = int(round(thr * width))
    out = []
    for i in range(width):
        if i == tpos:
            out.append(col("┃", C.YEL))
        elif i < fill:
            out.append("█")
        else:
            out.append(col("·", C.DIM))
    return "".join(out)


def print_one(name, model, proba, verdict, vec, short, thr, show_features=True):
    vcol = C.RED if verdict == "ИИ" else C.GRN
    print(f"\n{C.B}{name}{C.R}  {C.DIM}[модель {model}]{C.R}")
    print(f"  вердикт: {col(verdict.upper(), vcol)}   p(ИИ)={proba:.3f}  порог={thr:.2f}")
    print(f"  {prob_bar(proba, thr)}")
    if short:
        print(col(f"  [!] текст короче {MIN_RELIABLE_CHARS} симв — на коротком детектор "
                  "менее надёжен", C.YEL))
    if show_features:
        print(f"  {C.DIM}признаки v1 (для прозрачности):{C.R}")
        for n, v in zip(FEATURE_NAMES, vec):
            print(f"    {C.DIM}{n:16s}{C.R} {v:.3f}")


def read_text_file(path: Path) -> str:
    if path.suffix.lower() not in (".txt", ".md", ""):
        print(col(f"[i] {path.name}: ожидается .txt; читаю как plain text", C.DIM))
    return path.read_text(encoding="utf-8", errors="replace").strip()


def run_dir(model, d):
    d = Path(d)
    files = sorted(p for p in d.glob("*.txt"))
    if not files:
        sys.exit(col(f"[x] В {d} нет .txt файлов", C.RED))

    labels = {}
    lp = d / "labels.json"
    if lp.exists():
        labels = json.load(open(lp, encoding="utf-8"))

    rows = []
    for f in files:
        text = read_text_file(f)
        if not text:
            continue
        proba, thr, vec, short = score(model, text)
        verdict = "ИИ" if proba >= thr else "человек"
        print_one(f.name, model, proba, verdict, vec, short, thr, show_features=False)
        rows.append((f.name, proba, verdict, labels.get(f.name)))

    scored = [(p, v, gt) for (_, p, v, gt) in rows if gt is not None]
    if not scored:
        print(col("\n[i] labels.json нет — скоринг пропущен (показал только вердикты).", C.DIM))
        return

    n = len(scored)
    correct = sum(1 for p, v, gt in scored if (1 if v == "ИИ" else 0) == gt)
    hum = [(p, v) for p, v, gt in scored if gt == 0]
    ai = [(p, v) for p, v, gt in scored if gt == 1]
    fp = sum(1 for p, v in hum if v == "ИИ")
    tp = sum(1 for p, v in ai if v == "ИИ")

    print(f"\n{C.B}=== СКОРИНГ (модель {model}) ==={C.R}")
    print(f"  точность:        {correct}/{n} = {correct/n:.2%}")
    if hum:
        fpr = fp / len(hum)
        c = C.GRN if fpr <= 0.03 else (C.YEL if fpr <= 0.1 else C.RED)
        print(f"  FPR (люди→ИИ):   {col(f'{fpr:.2%}', c)}  ({fp}/{len(hum)} ложных обвинений)")
    if ai:
        rec = tp / len(ai)
        print(f"  recall (ИИ):     {rec:.2%}  ({tp}/{len(ai)} пойманных ИИ)")
    print(col("  Напоминание: вероятностная оценка, не доказательство.", C.DIM))


def main():
    ap = argparse.ArgumentParser(description="Детектор ИИ-текста (боевая модель v4)")
    ap.add_argument("--model", choices=["v1", "v2", "v3", "v4"], default="v4",
                    help="модель: v4 (боевая, по умолч.), v3 (прежняя боевая/откат), "
                         "v1 (признаки), v2 (tiny2-откат)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("file", nargs="?", help="путь к .txt файлу")
    g.add_argument("--text", help="текст прямо в аргументе")
    g.add_argument("--dir", help="папка с .txt (+ опц. labels.json)")
    args = ap.parse_args()

    if args.dir:
        run_dir(args.model, args.dir)
    elif args.text:
        proba, thr, vec, short = score(args.model, args.text)
        verdict = "ИИ" if proba >= thr else "человек"
        print_one("(--text)", args.model, proba, verdict, vec, short, thr)
    else:
        p = Path(args.file)
        if not p.exists():
            sys.exit(col(f"[x] Файл не найден: {p}", C.RED))
        text = read_text_file(p)
        if not text:
            sys.exit(col("[x] Файл пустой", C.RED))
        proba, thr, vec, short = score(args.model, text)
        verdict = "ИИ" if proba >= thr else "человек"
        print_one(p.name, args.model, proba, verdict, vec, short, thr)


if __name__ == "__main__":
    main()
