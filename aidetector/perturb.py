#!/usr/bin/env python3
"""
perturb.py — ФИКСИРОВАННЫЕ человекоподобные правки текста для ЗАМЕРА устойчивости
детектора к лёгкому редактированию. Инструмент ИЗМЕРЕНИЯ (аугментация теста), не обхода.

============================ ГРАНИЦА ПО ДИЗАЙНУ ============================
Модуль НИКОГДА не читает вероятность детектора и не зависит от него. Здесь НЕТ и
не должно появиться import features / joblib / model. Правки применяются ВСЛЕПУЮ,
по фиксированным правилам с фиксированным seed, один раз. Никакого цикла
«поправил → проверил p → подправил» и никакой оптимизации против модели. Если
такая обратная связь понадобится — это гуманизатор/обход, делать его нельзя
(см. CLAUDE.md). Только text -> text.
===========================================================================

Правки (как студент, слегка причёсывающий ИИ-черновик):
  1. дробим длинные предложения на короткие по сочинительным союзам;
  2. заменяем часть штампов на разговорные синонимы (фикс. словарь);
  3. вставляем разговорные связки в начало части предложений;
  4. лёгкая вариация порядка: вводное слово из начала в конец.
Детерминировано: seed фиксирован, выбор — от стабильного хэша текста.
"""

import argparse
import hashlib
import random
import re

from razdel import sentenize

DEFAULT_SEED = 12345

# вероятности правок ФИКСИРОВАНЫ (не подбираются под детектор)
P_CLICHE = 0.70
P_SPLIT = 0.60
P_CONNECTIVE = 0.30
P_REORDER = 0.50
LONG_SENT_WORDS = 16

# фиксированный словарь синонимов штампов (снижает «штампованность»)
SYNONYMS = {
    "важно отметить": "стоит сказать",
    "стоит отметить": "замечу",
    "следует отметить": "добавлю",
    "необходимо отметить": "скажу",
    "таким образом": "значит",
    "в заключение": "под конец",
    "играет важную роль": "много значит",
    "играет ключевую роль": "очень важна",
    "является неотъемлемой частью": "часть",
    "следует подчеркнуть": "подчеркну",
    "в современном мире": "сейчас",
    "в настоящее время": "сейчас",
    "не секрет, что": "понятно, что",
    "в первую очередь": "сперва",
    "необходимо учитывать": "надо помнить",
    "в целом можно сказать": "в общем",
    "подводя итог": "если коротко",
    "позволяет сделать вывод": "показывает",
    "открывает широкие возможности": "много чего даёт",
    "приобретают всё большую актуальность": "всё важнее",
}

CONNECTIVES = [
    "В общем,", "Честно говоря,", "По сути,", "На самом деле,",
    "Если что,", "Грубо говоря,", "Кстати,", "Короче,",
]

INTRO_WORDS = [
    "Однако", "Кроме того", "Таким образом", "Тем не менее",
    "Более того", "Впрочем", "Следовательно", "Конечно",
]

# правила дробления: (что ищем, чем начать второе предложение; "" -> capitalize тело)
SPLIT_RULES = [
    (", и ", ""),
    (", а также ", ""),
    (", а ", ""),
    (", но ", "Но "),
    (", однако ", "Однако "),
    (", поэтому ", "Поэтому "),
    (", что ", "Это "),
]


def _rng_for(text: str, seed: int) -> random.Random:
    """Детерминированный ГСЧ от стабильного хэша текста (md5 не зависит от запуска)."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed ^ h)


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def _replace_cliches(text: str, rng: random.Random) -> str:
    """Заменяем ЧАСТЬ штампов (каждое вхождение — с вероятностью P_CLICHE)."""
    for phrase, repl in SYNONYMS.items():
        def sub(m):
            if rng.random() >= P_CLICHE:
                return m.group(0)
            return _cap(repl) if m.group(0)[:1].isupper() else repl
        text = re.sub(re.escape(phrase), sub, text, flags=re.IGNORECASE)
    return text


def _split_long(s: str, rng: random.Random) -> str:
    """Длинное предложение дробим на два по первому подходящему союзу за серединой."""
    if len(s.split()) <= LONG_SENT_WORDS or rng.random() >= P_SPLIT:
        return s
    floor = int(len(s) * 0.35)
    pos, rule = -1, None
    for conn, head in SPLIT_RULES:
        i = s.find(conn, floor)
        if i != -1 and (pos == -1 or i < pos):
            pos, rule = i, (conn, head)
    if pos == -1:
        return s
    conn, head = rule
    left = s[:pos].rstrip(" ,")
    body = s[pos + len(conn):]
    right = (head + body) if head else _cap(body)
    return f"{left}. {right}"


def _insert_connective(s: str, rng: random.Random) -> str:
    """С вероятностью P_CONNECTIVE добавляем разговорную связку в начало."""
    if rng.random() >= P_CONNECTIVE:
        return s
    if any(s.startswith(c) for c in CONNECTIVES):
        return s
    return f"{rng.choice(CONNECTIVES)} {_lower_first(s)}"


def _reorder_intro(s: str, rng: random.Random) -> str:
    """Вводное слово из начала переносим в конец: «Однако, X.» -> «X, однако.»."""
    for w in INTRO_WORDS:
        if s.startswith(w + ",") or s.startswith(w + " ,"):
            if rng.random() >= P_REORDER:
                return s
            rest = _cap(s[len(w):].lstrip(" ,")).rstrip(".")
            return f"{rest}, {w.lower()}."
    return s


def perturb(text: str, seed: int = DEFAULT_SEED) -> str:
    """Один проход фиксированных правок ВСЛЕПУЮ. text+seed -> детерминированный результат."""
    rng = _rng_for(text, seed)
    text = _replace_cliches(text, rng)
    out = []
    for sent in sentenize(text):
        s = sent.text.strip()
        if not s:
            continue
        s = _split_long(s, rng)
        s = _insert_connective(s, rng)
        s = _reorder_intro(s, rng)
        out.append(s)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description="Фиксированные правки текста (замер устойчивости)")
    ap.add_argument("text")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    print(perturb(args.text, args.seed))


if __name__ == "__main__":
    main()
