#!/usr/bin/env python3
"""
genre_classify.py — ЭВРИСТИЧЕСКОЕ определение жанра документа (без ML) и рекомендация
модели (v3 / v4 / согласованность). НЕ трогает модели/пороги/обучение — это лёгкий
классификатор поверх существующего конвейера, чтобы подсказать пользователю, какой
скорер уместнее по жанру (известно: v4 завышает на гуманитарных академических текстах,
v3 — нет; на технических v4 сильнее).

Признаки берём с первых ~5000 символов СОДЕРЖАТЕЛЬНОГО текста (после отсева титула/
оглавления/таблиц тем же document_scan, что и детектор):
  - средняя длина предложения и её std (бёрстность);
  - доля абзацев с цитатами [Автор, год] / [N];
  - доля латиницы (иноязычные термины);
  - доля матем./программных символов;
  - плотность терминологических кластеров (гуманитарный / технический / научная статья);
  - плотность канцеляризмов (формальность).
"""

import re

from .document_scan import _WORD_RE, classify_sections, split_paragraphs

SAMPLE_CHARS = 5000

# Терминологические кластеры (леммы-префиксы; матчим по вхождению, регистронезависимо).
HUM_TERMS = re.compile(
    r"исследован|методологи|корпус|семантик|дискурс|лингвист|интерпретац|концепт|"
    r"нарратив|поэтик|философ|культур|социолог|историограф|художеств|литератур|"
    r"роман|произведени|жанр|метафор|стилист|герменевт|эстетик|идентичност", re.I)
TECH_TERMS = re.compile(
    r"систем|таблиц|интерфейс|база данных|\bбд\b|реализац|модул|сервер|алгоритм|"
    r"диаграмм|разработк|программн|функционал|приложени|пользовател|запрос|клиент|"
    r"архитектур|\bкод\b|развёртыван|тестирован|фреймворк|компонент|сущност", re.I)
SCI_TERMS = re.compile(
    r"гипотез|эксперимент|выборк|корреляц|значимост|респондент|статистическ|"
    r"регресс|p-значени|контрольн.{0,3} групп", re.I)
CANC = re.compile(
    r"обусловлен|представляет собой|в результате|таким образом|в связи с|"
    r"следует отметить|необходимо отметить|осуществляет|данн(?:ый|ая|ое|ых|ыми)\b", re.I)
# цитаты: [..1234..] (год в скобках) или [12], [3, 5] (номерные ссылки)
CITE = re.compile(r"\[[^\]]*\b\d{4}\b[^\]]*\]|\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
MATHCODE = re.compile(r"[=+\-→×÷≤≥∑∫√{}<>]|(?<!\w)(function|var|import|def|class|return|"
                      r"select|create|insert|public|void|int |float)(?!\w)", re.I)
SENT_SPLIT = re.compile(r"[.!?]+(?:\s|$)")


def _content_sample(text: str) -> tuple[str, list[str]]:
    """Первые ~SAMPLE_CHARS символов содержательной прозы + список её абзацев."""
    paras = split_paragraphs(text)
    split = classify_sections(paras)
    prose = [p.text for p in split.content
             if not p.text.startswith("⟦таблица") and len(p.text) >= 40]
    if not prose:                                   # запасной путь — любой контент
        prose = [p.text for p in split.content] or [text]
    out, acc = [], 0
    for p in prose:
        out.append(p)
        acc += len(p)
        if acc >= SAMPLE_CHARS:
            break
    return " ".join(out)[:SAMPLE_CHARS], out


def features(text: str) -> dict:
    sample, paras = _content_sample(text)
    words = _WORD_RE.findall(sample)
    nw = max(1, len(words))
    letters = [c for c in sample if c.isalpha()]
    nl = max(1, len(letters))
    sents = [s for s in SENT_SPLIT.split(sample) if s.strip()]
    slens = [len(_WORD_RE.findall(s)) for s in sents] or [0]
    import statistics as st
    mean_sl = sum(slens) / len(slens)
    std_sl = st.pstdev(slens) if len(slens) > 1 else 0.0
    cite_par = sum(1 for p in paras if CITE.search(p)) / max(1, len(paras))
    latin = sum(1 for c in letters if "a" <= c.lower() <= "z") / nl
    mathcode = len(MATHCODE.findall(sample)) / nw
    per1k = lambda rgx: len(rgx.findall(sample)) / nw * 1000
    return {
        "n_words": len(words), "n_paras": len(paras),
        "mean_sent_len": round(mean_sl, 1), "sent_std": round(std_sl, 1),
        "cite_frac": round(cite_par, 3), "latin_frac": round(latin, 3),
        "mathcode_per_w": round(mathcode, 3),
        "hum_per1k": round(per1k(HUM_TERMS), 1), "tech_per1k": round(per1k(TECH_TERMS), 1),
        "sci_per1k": round(per1k(SCI_TERMS), 1), "canc_per1k": round(per1k(CANC), 1),
    }


# --- агрегатные сигналы и решение (калибровано на 7 эталонах) ---
def _signals(f: dict) -> tuple[float, float]:
    """Технический и гуманитарный сигналы (сопоставимые шкалы)."""
    tech = f["tech_per1k"] + 800 * f["mathcode_per_w"] + 0.6 * f["sci_per1k"]
    hum = (f["hum_per1k"] + 120 * f["cite_frac"] + 80 * f["latin_frac"]
           + max(0, f["mean_sent_len"] - 14) * 1.2)
    return tech, hum


GENRE_TECH = "технический диплом/курсовая"
GENRE_HUM = "гуманитарный академический"
GENRE_MIXED = "смешанный/неявный"


def classify(text: str) -> dict:
    f = features(text)
    tech, hum = _signals(f)
    strong, weak = max(tech, hum), min(tech, hum)
    # «Смешанный» — если сигнала жанра почти нет (strong<8) ИЛИ оба сигнала сильны и
    # БЛИЗКИ по величине (weak/strong>0.75): признаки не дают чёткой картины. Порог
    # относительный, не абсолютный — иначе при высоких сигналах 6-балльный зазор
    # «случайно» решает жанр (digital humanities: много кода И много литведа → mixed).
    if strong < 8 or (weak > 8 and weak / strong > 0.75):
        genre, model, reason = (GENRE_MIXED, "agree",
            "признаки жанра нечёткие — рекомендую режим согласованности (a): красит абзацы "
            "по согласию v3 и v4, расхождения = зона жанрового перекоса")
        confident = False
    elif tech >= hum:
        genre, model, reason = (GENRE_TECH, "v4",
            "технический стиль (термины систем/БД/реализации, формулы) — v4 на этом жанре "
            "сильнее всего и без перекоса")
        confident = True
    else:
        genre, model, reason = (GENRE_HUM, "v3",
            "гуманитарный академический стиль (цитаты, латиница, длинные периоды) — здесь "
            "v4 ЗАВЫШАЕТ (стиль близок к обучающим ИИ-текстам), рекомендую v3")
        confident = True
    return {"genre": genre, "recommended": model, "reason": reason,
            "confident": confident, "tech_signal": round(tech, 1),
            "hum_signal": round(hum, 1), "features": f}


if __name__ == "__main__":
    import sys

    from .extract_text import extract_text
    t, _ = extract_text(sys.argv[1])
    r = classify(t)
    print(f"Жанр: {r['genre']} · рекомендую {r['recommended']}")
    print(f"  {r['reason']}")
    print(f"  tech={r['tech_signal']} hum={r['hum_signal']} | {r['features']}")
