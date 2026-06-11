#!/usr/bin/env python3
"""
document_scan.py — поабзацная карта «ИИ / человек» для одного документа (.txt).

ЗАЧЕМ отдельный модуль, а не прогон детектора по абзацам напрямую:
модель v1 надёжна только на тексте >=200 символов (на коротких FPR доходит до
0.24 — она начинает ложно обвинять). Половина реальных абзацев короче 200 симв,
поэтому гнать детектор «абзац -> вердикт» НЕЛЬЗЯ — это фабрика ложных обвинений.

Архитектура (важно): СЧИТАЕМ НА ОКНАХ, ПОКАЗЫВАЕМ НА АБЗАЦАХ.
  1. парсим абзацы (с их символьными позициями);
  2. по БЕЛОМУ списку заголовков оставляем только содержательную часть
     (Введение..Заключение + нумерованные главы), отбрасываем титул/оглавление/
     список литературы/приложения — и логируем, что отброшено;
  3. режем содержательную часть скользящим окном ~150 слов с перехлёстом ~50%
     (каждое окно гарантированно >200 симв — в зоне надёжности модели);
  4. гоним детектор (features.extract + model.joblib + порог из threshold_v1.json)
     по каждому ОКНУ -> p(ИИ) на окно;
  5. проецируем окна на абзацы: p(ИИ) абзаца = среднее p накрывающих окон,
     взвешенное по длине перекрытия. Абзац, на который НЕ приходится >=200 симв
     надёжного окна (слишком короткий) -> «недостаточно текста», а НЕ догадка;
  6. на выходе — КАРТА абзацев (p + метка) + сводка по долям, без единого вердикта
     на весь документ.

Модель/порог/features НЕ трогаем — только читаем (как сервис и detect_cli).
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib

from .extract_text import TABLE_MARK
from .features import FEATURE_NAMES, extract, to_vector
from .paths import model_path

# --- пути и пороги (совпадают с app.py / detect_cli.py) ----------------------
MODEL_PATH = model_path("model.joblib")
THRESHOLD_PATH = model_path("threshold_v1.json")
DEFAULT_THRESHOLD = 0.86            # порог «ИИ», калиброван под FPR<=3% (app.py)
MIN_RELIABLE_CHARS = 200           # ниже — модель v1 НЕнадёжна (app.py)

# --- параметры скользящего окна ---------------------------------------------
# Размер окна 150 слов (≈1000+ симв, надёжно >200) — «колено» кривой recall/FPR
# по замеру (measure_window.py). Было 300, но окно 300 слов размывает ИИ-вставку
# соседним человеческим текстом и почти не ловит её (recall на карте ~0.23 у v2);
# 75 слов ловит вдвое лучше, но FPR на людях прыгает к ~17% — окно перестаёт
# защищать человека от ложного обвинения. 150 — компромисс: recall к 300 примерно
# ×1.6 при умеренном росте ложных. На вердиктном пороге (0.86/0.90) вставки всё
# равно почти не ловятся — это ограничение порога/инфо-предела, не окна.
WINDOW_WORDS = 150                 # размер окна в словах
OVERLAP_RATIO = 0.5                # перехлёст соседних окон (50%)
# Шаг окна в словах. 50% перехлёст -> шаг = половина окна.
WINDOW_STEP = max(1, int(round(WINDOW_WORDS * (1 - OVERLAP_RATIO))))

# Нижняя граница «уверенно человек». Порог 0.86 — это «подозрительно».
# Между 0.5 и 0.86 модель склоняется к ИИ, но НЕ дотягивает до порога обвинения
# -> честная метка «неопределённо», а не «человек» и не «ИИ».
# 0.5 = равновероятно; ниже — модель скорее за человека.
UNCERTAIN_LOW = 0.5

# --- БЕЛЫЙ список заголовков (настраиваемый) ---------------------------------
# Бельый список надёжнее чёрного: мы явно описываем, ЧТО считать содержанием,
# а не пытаемся перечислить весь мусор. Сравнение — по нормализованному префиксу.

# Заголовки, открывающие содержательную часть.
CONTENT_START_MARKERS = ("введение",)
# Заголовки глав/разделов внутри содержания (тоже контент).
CHAPTER_MARKERS = ("глава", "раздел", "заключение")
# Нумерованный заголовок: «1. ...», «2.1 ...», «3.2.1. ...».
NUM_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S")
# Строка ОГЛАВЛЕНИЯ: текст, ТАБ или точки-лидеры, номер страницы в конце
# («Введение\t3», «СПИСОК ИСТОЧНИКОВ\t74», «1.1 Анализ … 7»). Это НЕ настоящий
# заголовок секции — иначе залатчимся на «СПИСОК ИСТОЧНИКОВ» из оглавления (а он
# встречается ДВАЖДЫ: в TOC и реальный в конце). Одиночный пробел перед числом НЕ
# берём — чтобы не спутать с «Глава 1» / «Раздел 3» (там число — номер главы).
_TOC_LINE_RE = re.compile(r"(?:\t|\.{2,})\s*\d{1,4}\s*$")
# Начало «хвоста» документа — всё отсюда и до конца выбрасываем.
BACKMATTER_MARKERS = (
    "список литературы", "список использованных источников",
    "использованная литература", "библиографический список",
    "библиография", "литература", "источники",
    "приложение", "приложения",
    "references", "bibliography",
)
# Оглавление/титульное (служебное, идёт ДО содержания).
TOC_MARKERS = ("оглавление", "содержание", "contents")

# Заголовок — короткая строка (иначе это обычный абзац, а не заголовок).
HEADING_MAX_WORDS = 9


# --- цвета (без зависимостей, как в detect_cli.py) ---------------------------
class C:
    R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GRN = "\033[32m"; YEL = "\033[33m"; CYAN = "\033[36m"


def col(s, c):
    return f"{c}{s}{C.R}"


# =============================================================================
# 1. ПАРСИНГ АБЗАЦЕВ
# =============================================================================
@dataclass
class Paragraph:
    idx: int          # порядковый индекс в ИСХОДНОМ документе (по всем абзацам)
    text: str
    start: int        # символьная позиция начала в исходном документе
    end: int          # символьная позиция конца (exclusive)
    kind: str = "?"   # "content" | "service:<тип>"
    # XML-метаданные из docx (если есть) — для УМНОЙ категоризации коротких абзацев.
    style: str = ""           # имя стиля (lower), напр. "heading 1"
    is_heading_xml: bool = False
    is_caption_xml: bool = False
    has_math: bool = False


def attach_block_meta(paras: list[Paragraph], blocks) -> bool:
    """Навешиваем XML-мета (стиль/подпись/формула) на Paragraph ПО ИНДЕКСУ.
    extract_blocks отдаёт blocks 1:1 с абзацами split_paragraphs, поэтому индекс
    совпадает. Защита: если длины не сходятся — НЕ навешиваем (падаем на эвристику),
    чтобы не приписать чужую разметку. Возвращает True, если навесили."""
    if not blocks or len(blocks) != len(paras):
        return False
    for p, b in zip(paras, blocks):
        p.style = b.get("style", "")
        p.is_heading_xml = bool(b.get("is_heading"))
        p.is_caption_xml = bool(b.get("is_caption"))
        p.has_math = bool(b.get("has_math"))
    return True


# Разделитель абзацев: одна или несколько пустых строк (\n\n, \n \n, и т.п.).
_PARA_SPLIT = re.compile(r"\n[ \t]*\n+")


def split_paragraphs(text: str) -> list[Paragraph]:
    """Режем текст на абзацы по пустым строкам, сохраняя символьные позиции."""
    paras: list[Paragraph] = []
    idx = 0
    pos = 0
    for chunk in _PARA_SPLIT.split(text):
        raw = chunk
        stripped = raw.strip()
        if stripped:
            # позиция стрипнутого текста внутри исходного документа
            start = text.find(stripped, pos)
            if start < 0:
                start = pos
            end = start + len(stripped)
            paras.append(Paragraph(idx=idx, text=stripped, start=start, end=end))
            idx += 1
            pos = end
        else:
            pos += len(raw)
    return paras


# =============================================================================
# 2. ОТСЕВ СЛУЖЕБНЫХ СЕКЦИЙ ПО БЕЛОМУ СПИСКУ
# =============================================================================
def _norm_heading(text: str) -> str:
    """Нормализуем строку для сравнения с маркерами: lower + срезаем нумерацию."""
    s = text.strip().lower()
    s = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", s)   # «1.2 Глава» -> «глава»
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_heading(text: str) -> bool:
    """Заголовок — короткая одиночная строка, а не абзац-простыня."""
    if "\n" in text.strip():
        return False
    return 1 <= len(text.split()) <= HEADING_MAX_WORDS


def _heading_kind(text: str) -> Optional[str]:
    """
    Тип заголовка по белому списку или None, если это не распознанный заголовок.
      content_start  — открывает содержание (Введение)
      content        — глава/раздел/Заключение/нумерованный заголовок
      backmatter     — список литературы / приложения (конец содержания)
      toc            — оглавление/содержание (служебное)
    """
    if not _looks_like_heading(text):
        return None
    if _TOC_LINE_RE.search(text):          # строка оглавления (…\t74) — не настоящий заголовок
        return "toc"
    n = _norm_heading(text)
    if any(n.startswith(m) for m in TOC_MARKERS):
        return "toc"
    if any(n.startswith(m) for m in BACKMATTER_MARKERS):
        return "backmatter"
    if any(n.startswith(m) for m in CONTENT_START_MARKERS):
        return "content_start"
    if any(n.startswith(m) for m in CHAPTER_MARKERS) or NUM_HEADING_RE.match(text):
        return "content"
    return None


@dataclass
class SectionSplit:
    content: list[Paragraph]
    dropped: list[tuple[Paragraph, str]] = field(default_factory=list)  # (абзац, причина)


# Точечные «лидеры» оглавления: «Введение .......... 3». Прозе они не свойственны,
# зато выдают строку TOC, даже если она вдруг оказалась длинной — телом не считаем.
_DOT_LEADER_RE = re.compile(r"\.{4,}")


def _is_table(text: str) -> bool:
    """Маркер таблицы из extract_text — данные, не авторская проза, в анализ не идут."""
    return text.startswith(TABLE_MARK)


def _is_body_prose(p: Paragraph) -> bool:
    """Длинный абзац НАСТОЯЩЕЙ прозы: >=MIN_RELIABLE_CHARS, без точек-лидеров TOC и не таблица."""
    return (len(p.text) >= MIN_RELIABLE_CHARS
            and not _DOT_LEADER_RE.search(p.text)
            and not _is_table(p.text))


def _is_chapter_heading(text: str) -> bool:
    """Заголовок ГЛАВЫ работы: Введение / Глава / Раздел / Заключение (без нумерации)."""
    n = _norm_heading(text)
    return (any(n.startswith(m) for m in CONTENT_START_MARKERS)
            or any(n.startswith(m) for m in CHAPTER_MARKERS))


def _body_continues_after(paras: list[Paragraph], kinds: list, idx: int) -> bool:
    """Есть ли ПОСЛЕ idx ещё ТЕЛО работы — заголовок главы (Введение/Глава/Раздел/
    Заключение), ВПЛОТНУЮ за которым идёт проза? Если да — back-matter тут НЕ настоящий
    (это пункт оглавления «СПИСОК ИСТОЧНИКОВ 74» или упоминание «Источники …» подзаголовком
    в середине — за ними ещё идут главы). Настоящий back-matter стоит ПОСЛЕ последней
    главы, и тела за ним уже нет. Нумерованные пункты-источники («1. ГОСТ …») главами
    НЕ считаются — они длинные (kind=None) и не матчат маркеры глав."""
    for j in range(idx + 1, len(paras) - 1):
        if _looks_like_heading(paras[j].text) and _is_chapter_heading(paras[j].text) \
                and _is_body_prose(paras[j + 1]):
            return True
    return False


def classify_sections(paras: list[Paragraph]) -> SectionSplit:
    """
    Оставляем содержательную часть: от настоящего content-заголовка до начала
    back-matter (список литературы/приложения). Всё до и всё после — служебное.

    Как находим НАСТОЯЩИЙ content_start (тут ломался отсев):
    в реальном Word-оглавлении каждый пункт — отдельный короткий абзац-заголовок
    («Введение», «Глава 1», «Список литературы»...), не отличимый от настоящего.
    Нельзя брать «первый заголовок-маркер» — залатчится на строку TOC. И нельзя
    брать «заголовок, за которым тело в пределах k абзацев»: на КОРОТКОМ оглавлении
    сам пункт TOC случайно попадает в пределы k от тела (рецидив на test_mixed).

    Устойчивый признак: настоящий заголовок стоит НЕПОСРЕДСТВЕННО (вплотную) перед
    первым длинным абзацем прозы. Строка оглавления так стоять НЕ может — сразу за
    ней идёт ДРУГАЯ короткая строка оглавления, а не проза. Признак не зависит от
    размера TOC (короткий / длинный раскладной / одним блоком): между оглавлением и
    телом ВСЕГДА стоит сам настоящий заголовок, поэтому ни один пункт TOC не
    оказывается вплотную перед длинным абзацем. Алгоритм: находим первый длинный
    абзац прозы и берём заголовок ровно перед ним.
    """
    kinds = [_heading_kind(p.text) for p in paras]
    n = len(paras)

    # Поиск тела начинаем ПОСЛЕ оглавления (Содержание/Оглавление), если оно есть.
    # ЗАЧЕМ: на титуле и в «Задании на ВКР» бывают длинные поля-абзацы (>200 симв:
    # «3. Исходные данные к работе: …», «4. Содержание расчётно-пояснительной записки…»),
    # которые проходят как проза и латчат first_body НА ФРОНТ-МАТЕР — ДО оглавления.
    # Тогда content_start уезжает в начало, а back_start цепляется за пункт оглавления
    # «СПИСОК ИСТОЧНИКОВ 74» -> в анализ попадают 26 абзацев титула/TOC (баг на реальном
    # ВКР). Реальное тело ВСЕГДА идёт ПОСЛЕ оглавления — оттуда и ищем первую прозу.
    toc_marker = next((i for i, k in enumerate(kinds) if k == "toc"), None)
    body_from = (toc_marker + 1) if toc_marker is not None else 0

    first_body = next((i for i in range(body_from, n) if _is_body_prose(paras[i])), None)
    # запасной путь: после оглавления прозы не нашли (или TOC нет) — ищем с начала
    if first_body is None:
        first_body = next((i for i, p in enumerate(paras) if _is_body_prose(p)), None)

    content_start = None
    if first_body is not None:
        prev = first_body - 1
        if prev >= 0 and kinds[prev] in ("content_start", "content"):
            content_start = prev          # заголовок ВПЛОТНУЮ перед прозой
        else:
            content_start = first_body     # перед прозой нет заголовка — тело с неё

    # запасной путь: длинной прозы нет вовсе (все абзацы короткие) — старая логика:
    # первый content-заголовок, а если их нет — весь текст как контент.
    if content_start is None:
        for i, k in enumerate(kinds):
            if k in ("content_start", "content"):
                content_start = i
                break
    if content_start is None:
        return SectionSplit(content=list(paras), dropped=[])

    # начало back-matter — СТРУКТУРНО: настоящий «Список литературы/Приложения» — это
    # заголовок, ПОСЛЕ которого тела работы уже нет (нет главы Введение/Глава/Раздел/
    # Заключение с прозой). Так отсекаем (а) пункт оглавления «СПИСОК ИСТОЧНИКОВ 74»
    # (за ним ещё идёт ВВЕДЕНИЕ + проза) и (б) упоминание «Источники …» подзаголовком
    # в середине (за ним ещё идут главы). Тот же принцип, что для content_start:
    # заголовок вплотную к характерному контенту, а не первое вхождение маркера.
    back_start = None
    for i in range(content_start + 1, n):
        if kinds[i] == "backmatter" and not _body_continues_after(paras, kinds, i):
            back_start = i
            break

    end = back_start if back_start is not None else n
    body_region = paras[content_start:end]
    # таблицы ВНУТРИ тела (схемы БД, расчёты) — данные, не проза: в анализ НЕ берём,
    # но показываем как служебное (не маскируем, что они были)
    content = [p for p in body_region if not _is_table(p.text)]
    tables_in_body = [p for p in body_region if _is_table(p.text)]

    # для лога различаем титул и оглавление: с какого абзаца пошёл TOC
    toc_idx = next((i for i in range(content_start) if kinds[i] == "toc"), None)

    dropped: list[tuple[Paragraph, str]] = []
    for i, p in enumerate(paras[:content_start]):
        if toc_idx is not None and i >= toc_idx:
            dropped.append((p, "оглавление (TOC)"))
        else:
            dropped.append((p, "титул (до содержательной части)"))
    for p in tables_in_body:
        dropped.append((p, "таблица (данные, не авторская проза)"))
    for p in paras[end:]:
        dropped.append((p, "после списка литературы (back-matter)"))

    for p in content:
        p.kind = "content"
    for p, _ in dropped:
        p.kind = "service"

    return SectionSplit(content=content, dropped=dropped)


# =============================================================================
# 3. НАРЕЗКА СКОЛЬЗЯЩИМ ОКНОМ
# =============================================================================
# Слово = токен, в котором есть хотя бы одна буква (число/пунктуацию не считаем
# «словом» при подсчёте до 300, но они остаются внутри подстроки окна).
_WORD_RE = re.compile(r"\w*[^\W\d_]\w*", re.UNICODE)


@dataclass
class Window:
    cstart: int       # позиция в «потоке содержания» (см. ниже)
    cend: int
    p_ai: float = -1.0


def _build_content_stream(content: list[Paragraph]):
    """
    Склеиваем содержательные абзацы в один поток через '\\n\\n' и запоминаем
    спан каждого абзаца В ЭТОМ ПОТОКЕ. Дальше вся оконная математика — в
    координатах потока (исходные позиции абзацев сохранены отдельно, для отчёта).
    """
    parts, spans = [], []
    pos = 0
    for p in content:
        spans.append((pos, pos + len(p.text)))
        parts.append(p.text)
        pos += len(p.text) + 2          # +2 за разделитель '\n\n'
    return "\n\n".join(parts), spans


def build_windows(stream: str) -> list[Window]:
    """
    Скользящее окно WINDOW_WORDS слов, шаг WINDOW_STEP (перехлёст ~50%).
    Хвост короче MIN_RELIABLE_CHARS присоединяем к предыдущему окну,
    а не оцениваем отдельно (иначе попадём в ненадёжную зону модели).
    """
    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(stream)]
    n = len(words)
    if n == 0:
        return []

    wins: list[Window] = []
    i = 0
    while True:
        j = min(i + WINDOW_WORDS, n)
        cstart = words[i][0]
        cend = words[j - 1][1]
        wins.append(Window(cstart=cstart, cend=cend))
        if j >= n:
            break
        i += WINDOW_STEP

    # хвост <200 симв -> вливаем в предыдущее окно
    if len(wins) >= 2 and (wins[-1].cend - wins[-1].cstart) < MIN_RELIABLE_CHARS:
        wins[-2].cend = wins[-1].cend
        wins.pop()

    return wins


# =============================================================================
# 4. ПРОГОН ДЕТЕКТОРА ПО ОКНАМ
# =============================================================================
def load_model():
    try:
        clf = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        sys.exit(col(f"[x] Не найден {MODEL_PATH}. Обучи модель (finalize_v1.py).", C.RED))
    thr = DEFAULT_THRESHOLD
    try:
        with open(THRESHOLD_PATH, encoding="utf-8") as f:
            thr = float(json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        print(col(f"[i] {THRESHOLD_PATH} не найден — беру дефолтный порог {thr}", C.DIM))
    return clf, thr


def score_windows(clf, stream: str, wins: list[Window]) -> None:
    """Каждое окно -> p(ИИ). Та же связка features.extract + predict_proba."""
    for w in wins:
        text = stream[w.cstart:w.cend]
        vec = to_vector(extract(text))
        w.p_ai = float(clf.predict_proba([vec])[0][1])


# =============================================================================
# 5. ПРОЕКЦИЯ ОКОН НА АБЗАЦЫ
# =============================================================================
def _union_len(intervals: list[tuple[int, int]]) -> int:
    """Длина объединения интервалов (без двойного счёта перехлёстов окон)."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + (ce - cs)


@dataclass
class ParaResult:
    idx: int
    text: str
    n_chars: int
    p_ai: Optional[float]      # None -> недостаточно текста
    label: str                 # человек | неопределённо | подозрительно | недостаточно текста
    covered_chars: int
    subcat: str = ""           # подкатегория «недостаточно текста» (см. classify_short)


# Подкатегории коротких абзацев, ушедших в «недостаточно текста». Цель — показать,
# ЧТО именно не попало в анализ: структура (заголовки/подписи/формулы) — это норма,
# а «короткая проза» — потенциальная лазейка для ИИ, нарезанного на мелкие тезисы.
SUBCAT_HEADING = "структурный заголовок"
SUBCAT_CAPTION = "подпись рис./табл."
SUBCAT_FORMULA = "формула/обозначение"
SUBCAT_LIST = "пункт списка"
SUBCAT_TABLE = "ячейка таблицы"
SUBCAT_PROSE = "короткая проза"

_CAP_TXT = re.compile(r"^\s*(рис(?:унок)?\.?|табл(?:ица)?\.?|схема|диаграмма|листинг|график)\s*№?\s*\d", re.I)
_NUM_HEAD = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s+\S")            # «1.1 …», «2.3.1 …» — подзаголовок
_BULLET = re.compile(r"^\s*([–—•·●*]|[a-zа-яё]\)|\(?\d+[.)])\s")
_FORMULA_SYM = re.compile(r"[=∑×÷≤≥≈∫√±]")
_CODE = re.compile(r"\b(def|return|self|import|class|elif|sum|append|print)\b|\w+\(\w*\)")


def classify_short(p: "Paragraph") -> str:
    """Подкатегория короткого абзаца («недостаточно текста»). Сначала XML-разметка
    docx (надёжнее), затем текстовые эвристики. «короткая проза» — то, что НЕ
    структура: реальный текст, просто нарезанный мельче 200 симв."""
    s = (p.text or "").strip()
    sl = s.lower()
    # 1) XML (если есть)
    if getattr(p, "is_heading_xml", False):
        return SUBCAT_HEADING
    if getattr(p, "is_caption_xml", False):
        return SUBCAT_CAPTION
    if getattr(p, "has_math", False):
        return SUBCAT_FORMULA
    # 2) текстовые эвристики
    if _CAP_TXT.match(s):
        return SUBCAT_CAPTION
    if _FORMULA_SYM.search(s) and len(s) < 175:
        return SUBCAT_FORMULA
    if _CODE.search(s) and ("(" in s or "=" in s or "." in s):
        return SUBCAT_FORMULA
    if sl.rstrip(":") in ("где", "примечание"):
        return SUBCAT_FORMULA
    nwords = len(_WORD_RE.findall(s))
    # заголовок: ключевые слова / нумерованный подзаголовок / короткий CAPS
    if (_is_chapter_heading(s) or _NUM_HEAD.match(s)
            or any(_norm_heading(s).startswith(m) for m in (TOC_MARKERS + BACKMATTER_MARKERS))):
        return SUBCAT_HEADING
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6 and len(s) < 70:
        return SUBCAT_HEADING
    if _BULLET.match(s):
        return SUBCAT_LIST
    # проза vs обрывок таблицы: предложение или достаточно слов -> проза
    if s.rstrip().endswith((".", "!", "?")) or nwords >= 8:
        return SUBCAT_PROSE
    if nwords <= 5:
        return SUBCAT_TABLE
    return SUBCAT_PROSE


def _label_for(p: float, thr: float) -> str:
    if p >= thr:
        return "подозрительно"
    if p >= UNCERTAIN_LOW:
        return "неопределённо"
    return "человек"


def project_to_paragraphs(content: list[Paragraph], spans, wins: list[Window],
                          thr: float) -> list[ParaResult]:
    """
    p(ИИ) абзаца = среднее p накрывающих окон, взвешенное по длине перекрытия.

    «Недостаточно текста»: если на абзац приходится <200 символов НАДЁЖНОГО окна
    (по сути — сам абзац короче 200 симв и не может «вместить» полноценное окно),
    мы НЕ присваиваем ему вердикт. Короткий кусок инерционно получил бы p соседнего
    окна — это и есть та самая догадка, которой мы избегаем.
    """
    results: list[ParaResult] = []
    for (ps, pe), para in zip(spans, content):
        plen = pe - ps
        weighted = 0.0
        wsum = 0.0
        covered_intervals: list[tuple[int, int]] = []
        for w in wins:
            os_, oe = max(ps, w.cstart), min(pe, w.cend)
            ov = oe - os_
            if ov > 0:
                weighted += w.p_ai * ov
                wsum += ov
                covered_intervals.append((os_, oe))
        covered = _union_len(covered_intervals)

        if covered < MIN_RELIABLE_CHARS:
            results.append(ParaResult(idx=para.idx, text=para.text, n_chars=plen,
                                     p_ai=None, label="недостаточно текста",
                                     covered_chars=covered, subcat=classify_short(para)))
        else:
            p = weighted / wsum
            results.append(ParaResult(idx=para.idx, text=para.text, n_chars=plen,
                                     p_ai=p, label=_label_for(p, thr),
                                     covered_chars=covered))
    return results


# =============================================================================
# ДИАГНОСТИКА: СКЛЕЙКА СОСЕДНЕЙ КОРОТКОЙ ПРОЗЫ
# =============================================================================
@dataclass
class GluedFragment:
    member_idxs: list[int]     # idx исходных абзацев, вошедших в склейку
    anchor_idx: int            # idx первого члена — место показа на карте
    text: str
    n_chars: int
    p_ai: Optional[float] = None   # заполняется ВНЕШНЕ (моделью), document_scan не скорит
    label: str = ""


def glue_short_prose(para_results: list[ParaResult]) -> list[GluedFragment]:
    """Соседние короткие ПРОЗАИЧЕСКИЕ абзацы («недостаточно текста», subcat=короткая
    проза), идущие подряд (idx без разрывов — между ними нет заголовка/таблицы/подписи),
    склеиваем в условный фрагмент. Если склейка >=2 абзацев и >=200 симв — её можно
    проанализировать (скорит вызывающий). Документ НЕ меняется — это только диагностика:
    видно, не спрятан ли ИИ в тексте, нарезанном на мелкие тезисы.

    Разрыв серии: любой НЕ-(короткая проза) абзац ИЛИ дырка в idx (значит между ними
    был отсеянный структурный элемент — таблица/картинка/служебное)."""
    frags: list[GluedFragment] = []
    run: list[ParaResult] = []

    def flush():
        if len(run) >= 2:
            txt = " ".join(r.text.strip() for r in run)
            if len(txt) >= MIN_RELIABLE_CHARS:
                frags.append(GluedFragment(member_idxs=[r.idx for r in run],
                                           anchor_idx=run[0].idx, text=txt, n_chars=len(txt)))

    last = None
    for r in para_results:
        is_prose_short = (r.p_ai is None and r.subcat == SUBCAT_PROSE)
        if is_prose_short and (last is None or r.idx == last + 1):
            run.append(r)
            last = r.idx
        else:
            flush()
            if is_prose_short:
                run = [r]
                last = r.idx
            else:
                run = []
                last = None
    flush()
    return frags


AGGR_TARGET_CHARS = 300        # целевой размер фрагмента агр.склейки (>=MIN_RELIABLE_CHARS)


def glue_short_prose_aggressive(para_results: list[ParaResult],
                                table_idxs=frozenset()) -> list[GluedFragment]:
    """АГРЕССИВНАЯ склейка (диагностика, отдельный режим): короткая проза объединяется
    ДАЖЕ через структурные разделители (заголовки/подписи/формулы/списки) И через длинные
    проанализированные абзацы — барьер ТОЛЬКО таблица (table_idxs между членами: слишком
    разные сущности). Накопив ~AGGR_TARGET_CHARS, закрываем фрагмент и начинаем новый —
    так покрываем МАКСИМУМ короткой прозы аналитически пригодными кусками (закрытие
    лазейки «нарезка на мелочь»). Документ не меняется; игнорирование границ может
    объединять РАЗНЫЕ смысловые блоки — это НЕ основной режим."""
    frags: list[GluedFragment] = []
    run: list[ParaResult] = []

    def flush():
        if len(run) >= 2:
            txt = " ".join(r.text.strip() for r in run)
            if len(txt) >= MIN_RELIABLE_CHARS:
                frags.append(GluedFragment(member_idxs=[r.idx for r in run],
                                           anchor_idx=run[0].idx, text=txt, n_chars=len(txt)))

    for r in para_results:
        if r.p_ai is None and r.subcat == SUBCAT_PROSE:
            if run and any(run[-1].idx < t < r.idx for t in table_idxs):
                flush()                    # таблица между членами -> барьер
                run = []
            run.append(r)
            if sum(len(x.text) for x in run) >= AGGR_TARGET_CHARS:
                flush()                    # достигли целевого размера -> отдельный фрагмент
                run = []
        # всё прочее (заголовки/подписи/формулы/списки/ДЛИННАЯ проза) -> ПЕРЕСЕКАЕМ
    flush()
    return frags


# =============================================================================
# ОРКЕСТРАЦИЯ
# =============================================================================
def scan_document(text: str, clf, thr: float, blocks=None) -> dict:
    paras = split_paragraphs(text)
    attach_block_meta(paras, blocks)          # XML-мета (если есть) — для подкатегорий
    split = classify_sections(paras)
    stream, spans = _build_content_stream(split.content)
    wins = build_windows(stream)
    if wins:
        score_windows(clf, stream, wins)
    para_results = project_to_paragraphs(split.content, spans, wins, thr)
    return {
        "threshold": thr,
        "n_paragraphs_total": len(paras),
        "dropped": split.dropped,
        "windows": wins,
        "paragraphs": para_results,
    }


def score_fragments(frags: list, score_fn, thr: float) -> None:
    """Скорит склеенные фрагменты переданной функцией score_fn(list[str])->list[float]
    и проставляет p_ai/label (по порогу карты thr). score_fn задаёт вызывающий —
    v1 (features+clf) или трансформер (sc.proba). document_scan моделей не выбирает."""
    if not frags:
        return
    ps = score_fn([f.text for f in frags])
    for f, p in zip(frags, ps):
        f.p_ai = float(p)
        f.label = _label_for(float(p), thr)


def predict_v1(clf, texts: list) -> list:
    """p(ИИ) для текстов моделью v1 (features+clf) — для склейки в v1-режиме."""
    return [float(clf.predict_proba([to_vector(extract(t))])[0][1]) for t in texts]


# =============================================================================
# 6. ВЫВОД — КАРТА, А НЕ КЛЕЙМО
# =============================================================================
_LABEL_COLOR = {
    "человек": C.GRN,
    "неопределённо": C.YEL,
    "подозрительно": C.RED,
    "недостаточно текста": C.DIM,
}


def _preview(text: str, n: int = 70) -> str:
    one = re.sub(r"\s+", " ", text).strip()
    return one if len(one) <= n else one[:n - 1] + "…"


def format_report(res: dict) -> str:
    out: list[str] = []
    thr = res["threshold"]

    # --- лог отброшенных служебных секций ---
    out.append(f"\n{C.B}=== ОТСЕВ СЛУЖЕБНЫХ СЕКЦИЙ (белый список) ==={C.R}")
    if res["dropped"]:
        for p, reason in res["dropped"]:
            out.append(col(f"  [отброшено] абз.{p.idx:>2}: «{_preview(p.text, 50)}» "
                          f"— {reason}", C.DIM))
    else:
        out.append(col("  ничего не отброшено (структурных заголовков не найдено — "
                      "анализирую весь текст)", C.DIM))

    # --- окна ---
    wins = res["windows"]
    out.append(f"\n{C.B}=== ОКНА ({WINDOW_WORDS} слов, перехлёст {int(OVERLAP_RATIO*100)}%) ==={C.R}")
    if not wins:
        out.append(col("  содержательного текста не нашлось", C.RED))
    for k, w in enumerate(wins):
        out.append(f"  окно {k+1}: симв.{w.cstart}–{w.cend} "
                  f"({w.cend - w.cstart} симв)  p(ИИ)={w.p_ai:.3f}")

    # --- карта абзацев ---
    out.append(f"\n{C.B}=== КАРТА АБЗАЦЕВ (порог ИИ={thr:.2f}) ==={C.R}")
    for r in res["paragraphs"]:
        lc = _LABEL_COLOR.get(r.label, C.R)
        p_txt = "  —  " if r.p_ai is None else f"p(ИИ)={r.p_ai:.3f}"
        out.append(f"  абз.{r.idx:>2} [{col(r.label.upper(), lc)}] {p_txt}  "
                  f"{C.DIM}({r.n_chars} симв){C.R}")
        out.append(f"        {C.DIM}{_preview(r.text)}{C.R}")

    # --- сводка по долям проанализированного текста ---
    paras = res["paragraphs"]
    total_chars = sum(r.n_chars for r in paras) or 1
    cats = ["подозрительно", "неопределённо", "человек", "недостаточно текста"]
    out.append(f"\n{C.B}=== СВОДКА (доля проанализированного текста) ==={C.R}")
    for cat in cats:
        chars = sum(r.n_chars for r in paras if r.label == cat)
        n = sum(1 for r in paras if r.label == cat)
        if chars == 0:
            continue
        lc = _LABEL_COLOR.get(cat, C.R)
        bar = "█" * int(round(chars / total_chars * 24))
        label_cell = f"{cat:<20}"
        out.append(f"  {col(label_cell, lc)} {chars/total_chars:5.1%}  "
                  f"{lc}{bar}{C.R}  ({n} абз., {chars} симв)")

    # --- обязательный дисклеймер ---
    out.append("")
    out.append(col("  Это вероятностная оценка ПО ФРАГМЕНТАМ, не доказательство авторства.",
                  C.YEL))
    out.append(col("  «Подозрительно» = повод присмотреться, а НЕ вердикт. Один общий ярлык",
                  C.DIM))
    out.append(col("  на работу не выдаём — только карту. Решения о санкциях по детектору "
                  "недопустимы.", C.DIM))
    return "\n".join(out)


def main():
    # Переиспользуем единый парсер: .txt/.docx/.pdf — никакого дублирования логики.
    from .extract_text import extract_text

    ap = argparse.ArgumentParser(
        description="Поабзацная карта ИИ/человек для документа (.txt/.docx/.pdf, детектор v1)")
    ap.add_argument("file", help="путь к документу (.txt, .docx или .pdf)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(col(f"[x] Файл не найден: {path}", C.RED))

    # docx/pdf -> чистый текст ПЕРЕД конвейером; .txt extract_text отдаёт как есть.
    text, warnings = extract_text(path)
    for w in warnings:
        print(col(f"[!] {w}", C.YEL))
    if not text.strip():
        sys.exit(col("[x] Текста для анализа нет (см. предупреждение выше). "
                    "Сканы требуют OCR — это отдельный этап.", C.RED))

    clf, thr = load_model()
    res = scan_document(text, clf, thr)
    print(f"\n{C.B}Документ:{C.R} {path.name}  "
          f"({res['n_paragraphs_total']} абзацев всего)")
    print(format_report(res))


if __name__ == "__main__":
    main()
