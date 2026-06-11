#!/usr/bin/env python3
"""
measure_window.py — ЧАСТЬ Б: размер окна скана vs размен recall/FPR.

ЗАЧЕМ. Скан режет документ скользящим окном (по умолчанию 300 слов) и красит
абзацы. Большое окно «размывает» короткую ИИ-вставку соседним человеческим
текстом -> вставка тонет (низкий recall). Маленькое окно лучше локализует
вставку, но каждое окно — короче, p(ИИ) на нём шумнее -> рискуем ложно обвинить
человека (выше FPR). Меряем этот размен НА ОБЕИХ моделях (v1 признаки / v2
трансформер) для окон 300/150/75 слов.

ЧТО МЕРЯЕМ.
  recall = доля ИИ-вставок, помеченных «подозрительно» (p(ИИ) абзаца >= порога);
  FPR    = доля ЧЕЛОВЕЧЕСКИХ абзацев, помеченных «подозрительно».
Абзац короче надёжной зоны -> «недостаточно текста» (без вердикта) — для recall
считается НЕ пойманным (продукт его не флагует), для FPR не учитывается.

ДАННЫЕ. (1) Синтетический батч документов из РАЗНЫХ размеченных абзацев
data/test.jsonl (без повторов — повтор сам по себе читается как ИИ, см.
build_synthetic.py): mixed-документы (вперемешку человек/ИИ) дают recall и часть
человеческих абзацев; diploma-документы (только человек) дают FPR.
(2) Реальные samples/test_mixed.docx и samples/test_diploma.docx — якорь на живых
документах (ground truth из samples/test_key.md).

Конвейер скана НЕ меняем: переиспользуем функции document_scan, варьируем только
ds.WINDOW_WORDS/WINDOW_STEP и источник p(ИИ) окна (v1 признаки / v2 трансформер) —
ровно как detector_tui подменяет скорер. Ничего не дообучаем.
"""

import json
import random
import re
from pathlib import Path

import joblib
import numpy as np

from aidetector import document_scan as ds
from aidetector import transformer_score as TF
from aidetector.extract_text import extract_text
from aidetector.paths import model_path
from aidetector.featcache import vectors as feat_vectors

WINDOWS = [300, 150, 75]
THR = {"v1": 0.86, "v2": 0.90}        # боевые (вердиктные) пороги — «обвинение»
THR_HL = {"v1": 0.60, "v2": 0.55}     # мягкие пороги ПОДСВЕТКИ карты в TUI (для справки)
CLF = joblib.load(model_path("model.joblib"))     # v1


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ============================ скоринг окон ===================================
def _score_v1(stream, wins):
    texts = [stream[w.cstart:w.cend] for w in wins]
    if not texts:
        return
    V = np.array(feat_vectors(texts), dtype=np.float32)   # кэш фич на диске
    P = CLF.predict_proba(V)[:, 1]
    for w, p in zip(wins, P):
        w.p_ai = float(p)


def _score_v2(stream, wins):
    texts = [stream[w.cstart:w.cend] for w in wins]
    if not texts:
        return
    for w, p in zip(wins, TF.proba(texts)):
        w.p_ai = float(p)


def scan(text: str, model: str, W: int, thr: float):
    """Скан с окном W слов выбранной моделью -> para_results (метки по thr)."""
    ds.WINDOW_WORDS = W
    ds.WINDOW_STEP = max(1, int(round(W * (1 - ds.OVERLAP_RATIO))))   # перехлёст 50%
    paras = ds.split_paragraphs(text)
    split = ds.classify_sections(paras)
    stream, spans = ds._build_content_stream(split.content)
    wins = ds.build_windows(stream)
    if wins:
        (_score_v2 if model == "v2" else _score_v1)(stream, wins)
    return ds.project_to_paragraphs(split.content, spans, wins, thr)


# ============================ синтетический батч =============================
TITLE = [
    "Министерство науки и высшего образования Российской Федерации",
    "Национальный исследовательский технический университет",
    "ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА\nна тему: «Обнаружение сетевых вторжений»",
    "Выполнил: студент группы ИБ-41\nПроверил: доц. Кравцов М. В.",
    "Москва — 2021",
]
TOC = ["Содержание",
       "Введение .......................... 3\n"
       "1. Анализ подходов ................ 5\n"
       "2. Практическая реализация ........ 12\n"
       "Заключение ........................ 20\n"
       "Список литературы ................. 22"]
BACK = ["Список литературы",
        "1. Олифер В. Г. Компьютерные сети. — СПб.: Питер, 2020. — 100 с.",
        "2. Bishop M. Computer Security. — Addison-Wesley, 2018. — 200 с.",
        "Приложение А", "Сводные результаты эксперимента приведены в таблице."]


def build_doc(items):
    """items: list[(label, text_flat)] -> (doc_text, {text_flat: label}).
    Размещаем по секциям (Введение/Глава1/Глава2/Заключение), вставляем заголовки.
    Заголовки в label_map НЕ попадают -> при сверке отсеются сами."""
    n = len(items)
    a = 2
    b = a + (n - 3 + 1) // 2
    secs = [("Введение", items[:a]),
            ("1. Анализ подходов", items[a:b]),
            ("2. Практическая реализация", items[b:n - 1]),
            ("Заключение", items[n - 1:])]
    parts = list(TITLE) + list(TOC)
    for head, sec in secs:
        parts.append(head)
        parts.extend(txt for _, txt in sec)
    parts += BACK
    label_map = {txt: lbl for lbl, txt in items}
    return "\n\n".join(parts) + "\n", label_map


def make_batch(seed=42):
    rows = load("data/test.jsonl")
    hum = [flat(r["text"]) for r in rows if r["label"] == 0 and len(r["text"]) >= 250]
    ai = [flat(r["text"]) for r in rows if r["label"] == 1 and len(r["text"]) >= 250]
    rng = random.Random(seed)
    rng.shuffle(hum)
    rng.shuffle(ai)
    hi = ai_i = 0
    docs = []   # (kind, doc_text, label_map)

    # mixed: 12 док × 8 абзацев, паттерн H A H A A H A A  (5 ИИ + 3 человек)
    pattern = [0, 1, 0, 1, 1, 0, 1, 1]
    for _ in range(12):
        items = []
        for lbl in pattern:
            if lbl == 1:
                items.append((1, ai[ai_i])); ai_i += 1
            else:
                items.append((0, hum[hi])); hi += 1
        doc, lm = build_doc(items)
        docs.append(("mixed", doc, lm))

    # diploma: 10 док × 6 человеческих абзацев
    for _ in range(10):
        items = [(0, hum[hi + k]) for k in range(6)]
        hi += 6
        doc, lm = build_doc(items)
        docs.append(("diploma", doc, lm))

    print(f"батч: mixed=12  diploma=10  | использовано human={hi}/{len(hum)}  ai={ai_i}/{len(ai)}")
    return docs


# ============================ агрегация по батчу =============================
def eval_isolated(docs, model, thr):
    """Эталон «идеальной локализации»: каждый абзац скорится В ОДИНОЧКУ (окно = сам
    абзац, ~50 слов, без соседей-размывателей). Это потолок recall, к которому
    оконный скан может лишь приближаться; разрыв с ним = цена размывания окном."""
    ai_texts, hum_texts = [], []
    for kind, doc, lm in docs:
        for txt, lbl in lm.items():
            (ai_texts if lbl == 1 else hum_texts).append(txt)
    if model == "v2":
        pa = TF.proba(ai_texts); ph = TF.proba(hum_texts)
    else:
        pa = CLF.predict_proba(np.array(feat_vectors(ai_texts), dtype=np.float32))[:, 1]
        ph = CLF.predict_proba(np.array(feat_vectors(hum_texts), dtype=np.float32))[:, 1]
    pa, ph = np.asarray(pa), np.asarray(ph)
    return dict(recall=float((pa >= thr).mean()), fpr=float((ph >= thr).mean()),
                ai_total=len(pa), ai_flag=int((pa >= thr).sum()), ai_insuff=0,
                hum_total=len(ph), hum_flag=int((ph >= thr).sum()))


def eval_batch(docs, model, W, thr):
    """-> dict с recall (ИИ-вставки) и FPR (человеческие абзацы) по всему батчу."""
    ai_total = ai_flag = ai_insuff = 0
    hum_total = hum_flag = 0
    for kind, doc, lm in docs:
        for r in scan(doc, model, W, thr):
            lbl = lm.get(r.text)
            if lbl is None:
                continue                       # заголовок/служебное — не наш абзац
            flagged = (r.label == "подозрительно")
            if lbl == 1:
                ai_total += 1
                if r.p_ai is None:
                    ai_insuff += 1
                if flagged:
                    ai_flag += 1
            else:
                hum_total += 1
                if flagged:
                    hum_flag += 1
    return dict(recall=ai_flag / max(ai_total, 1), fpr=hum_flag / max(hum_total, 1),
                ai_total=ai_total, ai_flag=ai_flag, ai_insuff=ai_insuff,
                hum_total=hum_total, hum_flag=hum_flag)


# ============================ реальные документы =============================
# AI-вставки в test_mixed.docx (по ключу samples/test_key.md): два длинных + один
# короткий. Опознаём по префиксу абзаца.
MIXED_AI_PREFIXES = {
    "В современном мире обеспечение информационной безопасности": "long",
    "Машинное обучение является эффективным инструментом": "short",
    "Анализ существующих подходов позволяет сделать вывод": "long",
}


def _ai_kind(text: str):
    for pref, kind in MIXED_AI_PREFIXES.items():
        if text.startswith(pref):
            return kind
    return None


def eval_real_mixed(model, W, thr):
    text, _ = extract_text(Path("samples/test_mixed.docx"))
    res = scan(text, model, W, thr)
    long_total = long_flag = short_flag = short_total = 0
    hum_total = hum_flag = 0
    for r in res:
        kind = _ai_kind(r.text)
        flagged = (r.label == "подозрительно")
        if kind == "long":
            long_total += 1; long_flag += int(flagged)
        elif kind == "short":
            short_total += 1; short_flag += int(flagged)
        elif r.p_ai is not None:            # получил вердикт и это не ИИ-вставка -> человек
            hum_total += 1; hum_flag += int(flagged)
    return dict(ai_long=f"{long_flag}/{long_total}", ai_short=f"{short_flag}/{short_total}",
                hum_fp=f"{hum_flag}/{hum_total}")


def eval_real_diploma(model, W, thr):
    text, _ = extract_text(Path("samples/test_diploma.docx"))
    res = scan(text, model, W, thr)
    tot = flag = 0
    for r in res:
        if r.p_ai is not None:              # абзац с вердиктом; в дипломе все — люди
            tot += 1; flag += int(r.label == "подозрительно")
    return dict(hum_fp=f"{flag}/{tot}", fpr=flag / max(tot, 1))


# ================================== main =====================================
def main():
    docs = make_batch()
    out = {"verdict_thr": THR, "highlight_thr": THR_HL, "batch": {}, "real": {}}

    for thr_name, THRS in [("ВЕРДИКТНЫЙ (обвинение)", THR), ("ПОДСВЕТКА карты TUI", THR_HL)]:
        print("\n" + "=" * 78)
        print(f"  БАТЧ — порог: {thr_name}   v1={THRS['v1']}  v2={THRS['v2']}")
        print("=" * 78)
        print(f"  {'окно':>5} {'модель':>6} | {'recall ИИ':>20} | {'FPR люди':>18} | "
              f"{'ИИ«недост.текста»':>16}")
        # эталон-потолок: абзац в одиночку (окно = сам абзац, без размывания)
        for model in ("v1", "v2"):
            r = eval_isolated(docs, model, THRS[model])
            rec = f"{r['recall']:.3f} ({r['ai_flag']}/{r['ai_total']})"
            fpr = f"{r['fpr']:.3f} ({r['hum_flag']}/{r['hum_total']})"
            print(f"  {'абзац':>5} {model:>6} | {rec:>20} | {fpr:>18} | {r['ai_insuff']:>16}")
            out["batch"][f"{thr_name}|isolated|{model}"] = r
        for W in WINDOWS:
            for model in ("v1", "v2"):
                r = eval_batch(docs, model, W, THRS[model])
                rec = f"{r['recall']:.3f} ({r['ai_flag']}/{r['ai_total']})"
                fpr = f"{r['fpr']:.3f} ({r['hum_flag']}/{r['hum_total']})"
                print(f"  {W:>5} {model:>6} | {rec:>20} | {fpr:>18} | {r['ai_insuff']:>16}")
                out["batch"][f"{thr_name}|{W}|{model}"] = r

    # --- реальные документы (якорь) — на ВЕРДИКТНОМ пороге ---
    print("\n" + "=" * 78)
    print("  РЕАЛЬНЫЕ ДОКУМЕНТЫ (вердиктный порог) — якорь на живых файлах")
    print("=" * 78)
    print("  test_mixed.docx: ИИ-вставки 2 длинных + 1 короткая (ключ test_key.md)")
    print(f"  {'окно':>5} {'модель':>6} | {'ИИ длинные':>12} {'ИИ короткая':>12} {'ложн.люди':>12}")
    for W in WINDOWS:
        for model in ("v1", "v2"):
            m = eval_real_mixed(model, W, THR[model])
            print(f"  {W:>5} {model:>6} | {m['ai_long']:>12} {m['ai_short']:>12} {m['hum_fp']:>12}")
            out["real"][f"mixed|{W}|{model}"] = m
    print("\n  test_diploma.docx: 100% человек — любое «подозрительно» = ложное")
    print(f"  {'окно':>5} {'модель':>6} | {'ложн.абзацы':>12} {'FPR':>8}")
    for W in WINDOWS:
        for model in ("v1", "v2"):
            d = eval_real_diploma(model, W, THR[model])
            print(f"  {W:>5} {model:>6} | {d['hum_fp']:>12} {d['fpr']:>8.3f}")
            out["real"][f"diploma|{W}|{model}"] = d

    json.dump(out, open("data/partB_window.json", "w"), ensure_ascii=False, indent=2)
    print("\n[сохранено] data/partB_window.json")


if __name__ == "__main__":
    main()
