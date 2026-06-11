#!/usr/bin/env python3
"""
detector_tui.py — терминальный TUI для проверки документов (Textual).

Это ТОЛЬКО UI-оболочка поверх готового бэкенда. ML не трогаем:
  extract_text.extract_text  — достать чистый текст из .docx/.pdf/.txt;
  document_scan.*            — поабзацная карта «ИИ/человек» (окна 150сл/50%,
                               отсев служебных секций, метка «недостаточно текста»);
  model.joblib / threshold_v1.json (v1, признаки), v2_model/ (v2, tiny2) и
  v3_model/ + threshold_v3.json (v3, ruBert-base — БОЕВАЯ, дефолт) — модели и пороги.

Переключение модели (v1→v2→v3→v4, клавиша m) — это подмена ТОЛЬКО скорера окна: остальной конвейер
(нарезка, отсев, проекция на абзацы, правило «недостаточно текста») у обеих моделей
один и тот же — функции document_scan переиспользуются как есть, файл не правится.

При ОТКРЫТИИ документа сразу определяется ЖАНР (genre_classify, эвристика <1с) и
авто-выбирается модель: технический → v4, гуманитарный академический → v3 (там v4
завышает), смешанный → база v4 + сразу режим согласованности. Рекомендация и причина
показаны в шапке сводки; клавиша m переопределяет вручную.

Клавиши в отчёте:
  m — модель v1→v2→v3→v4 (переопределить авто-выбор по жанру) · o — открыть файл · q — выход
  g — ДИАГНОСТИКА «склейка короткой прозы»: соседние короткие прозаические абзацы
      (<200 симв, не разделённые заголовком/таблицей/подписью) склеиваются и
      проверяются — видно, не спрятан ли ИИ в тексте, нарезанном на мелкие тезисы.
      Карта показывает склейки, а в СВОДКУ добавляется вторая строка «с учётом
      склейки»: +X подозрительно / +Y неопределённо и общая доля подозрительного.
      Основную сводку-факт НЕ подменяет; повторный g — выключить. Документ не меняется.
  G — АГРЕССИВНАЯ склейка (заглавная): то же, но ЧЕРЕЗ структурные границы
      (заголовки/подписи/формулы), барьер — только таблицы. Покрывает максимум короткой
      прозы (закрытие лазейки нарезки между структурными элементами). Фиолетовая рамка
      «⊗ агр. склейка», третья строка сводки. ВЫКЛ по умолчанию — может объединять разные
      смысловые блоки, НЕ основной режим. Повторный G — выключить.
  a — СОГЛАСОВАННОСТЬ v3+v4: документ скорится ОБЕИМИ моделями, абзацы красятся по
      согласию — красный (оба видят ИИ), зелёный (оба не видят), ОРАНЖЕВЫЙ (расхождение).
      Расхождение = жанр-чувствительная зона: на гуманитарных/академических работах v4
      склонен завышать (стиль близок к обучающим текстам), v3 — нет. Делает ограничение
      v4 видимым сигналом «перепроверить», а не прячет его. Повторный a — выключить.
  e — ЭКСПОРТ отчёта рядом с документом: <имя>.report.md и .report.html
      (HTML — с цветной подсветкой абзацев и якорями).
Сводка раскладывает «недостаточно текста» на подкатегории по XML docx (заголовок/
подпись/формула/ячейка) и эвристике — видно, ЧТО именно не попало в анализ.

Запуск:
  python detector_tui.py                 # TUI, дерево стартует из samples/
  python detector_tui.py <папка|файл>    # дерево из папки / сразу открыть файл
  python detector_tui.py --text F.docx --model v4   # headless-отчёт (для CI/проверки)
  python detector_tui.py --export F.docx            # headless: сохранить .md + .html
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from aidetector import document_scan as ds
from aidetector import genre_classify
from aidetector.extract_text import extract_blocks, extract_text

# =============================================================================
# ЯДРО (без UI): файл -> Report. Переиспользуется и TUI, и headless-режимом.
# =============================================================================
EXTS = {".docx", ".pdf", ".txt"}
MODELS = ["v1", "v2", "v3", "v4"]            # порядок переключения по клавише m
MODEL_SHORT = {"v1": "v1 · признаки", "v2": "v2 · tiny2", "v3": "v3 · ruBert-base",
               "v4": "v4 · ruBert-base+"}
MODEL_FULL = {
    "v1": "v1 · признаки (perplexity rugpt + GLTR rank-entropy), консервативный, низкий FPR",
    "v2": "v2 · трансформер rubert-tiny2 (29M) — лёгкий откат",
    "v3": "v3 · трансформер ruBert-base (178M) — прежняя боевая, консервативнее (откат)",
    "v4": "v4 · ruBert-base, переобучен с ИИ Llama3.1/Mistral — БОЕВАЯ: шире покрытие генераторов",
}

# =============================================================================
# ДВА РАЗНЫХ ПОРОГА — НЕ ПУТАТЬ.
#
#   1) ВЕРДИКТНЫЙ порог (threshold_v1.json=0.86 / threshold_v2.json=0.90) —
#      FPR-калиброван (≤3% ложных обвинений). Это БОЕВОЙ порог моделей для
#      вердикта по тексту (сервис app.py, detect_cli). Здесь мы его НЕ трогаем
#      и для раскраски карты НЕ применяем — только показываем для контекста.
#
#   2) Порог ПОДСВЕТКИ карты (HIGHLIGHT_THR ниже) — ТОЛЬКО для цвета абзаца в
#      этом TUI. У карты другая цена ошибки: «ложно подсветить» дёшево (глаз
#      перепроверит), «не подсветить ИИ» дорого (ради подсветки карта и нужна)
#      → порог МЯГЧЕ вердиктного. Влияет ИСКЛЮЧИТЕЛЬНО на цвет в TUI; модели,
#      их FPR и вердиктные пороги не меняются.
#
# Подобран по p(ИИ) абзацев тестовых docx (p считается на окнах, от порога не
# зависит): чистый ИИ должен краснеть, человек — оставаться зелёным.
# ВАЖНО: порог карты — ПОВЕРСИОННЫЙ, НЕ наследуется. У каждой модели своё
# распределение p, поэтому одно и то же число даёт разную картину.
#   v1: человек(diploma) max p=0.15 vs ИИ(pure_ai) p=0.97 — огромный зазор;
#       0.60 (мягче вердиктного 0.86) надёжно краснит ИИ, человек зелёный.
#   v2: ИИ p≈0.556 ПЕРЕКРЫВАЕТСЯ с верхом человека (0.51–0.572) — чистого
#       разделения на уровне абзаца нет; 0.55 краснит pure_ai ценой ~2 ложных
#       красных в diploma.
#   v3: ЧИСТОЕ разделение — diploma max p=0.46 vs pure_ai min p=0.97, перекрытия
#       НЕТ (большая ёмкость убрала проблему v2). 0.60 краснит весь pure_ai при 0
#       ложных в diploma.
#   v4: распределение p СДВИНУТО ВВЕРХ относительно v3 (менее консервативна) →
#       0.60 наследовать НЕЛЬЗЯ (шумит). Замер по абзацам docx:
#         чистый человек test_diploma: max p=0.30 (q90=0.26) — весь зелёный;
#         чистый ИИ test_pure_ai: min p=0.98, медиана 0.99;
#         причёсанный ИИ test_pure_ai_ghost: медиана 0.79 (потолок порога!);
#         реальный ВКР (человек, шумный): медиана 0.39, q90 0.66.
#       Порог 0.72: pure_ai 5/5 и причёсанный ИИ 4/5 (медиана 0.79>0.72) красны,
#       test_diploma 0 ложных, на реальном ВКР зоны внимания 24->7 (остаются
#       плотные шаблонные кластеры, уходит единичный шум 0.60–0.72). Выше 0.79
#       нельзя — теряем причёсанный ИИ (при 0.80 ghost 2/5).
HIGHLIGHT_THR = {"v1": 0.60, "v2": 0.55, "v3": 0.60, "v4": 0.72}

# Второй слой автовыбора: если жанр сказал v4, но v3/v4 СИЛЬНО расходятся (v4 видит ИИ
# там, где v3 УВЕРЕННО человек) — не доверять v4 слепо, уйти в согласованность.
# «Сильное» расхождение, а не любое: на 1.docx v3 тоже склоняется к ИИ (просто ниже
# порога) — это не перекос; на Антик1 v3 уверенно человек (p<0.45) — это перекос v4.
AGREE_DIV_THRESH = 0.15        # доля сильно-расходящихся окон → автосогласованность
AGREE_V3_HUMAN = 0.45          # v3 «уверенно человек» ниже этого

# label из document_scan -> (css-класс, цвет-markup, человекочитаемое имя)
STYLE = {
    "подозрительно":      ("v-ai",           "red",    "ИИ — ПОДОЗРИТЕЛЬНО"),
    "неопределённо":      ("v-uncertain",    "yellow", "НЕОПРЕДЕЛЁННО"),
    "человек":            ("v-human",        "green",  "ЧЕЛОВЕК"),
    "недостаточно текста": ("v-insufficient", "bright_black", "НЕДОСТАТОЧНО ТЕКСТА"),
}
CATS = ["подозрительно", "неопределённо", "человек", "недостаточно текста"]

# Режим «согласованность v3+v4»: метка абзаца -> (css-класс, цвет-markup, имя).
# Расхождение (один видит ИИ, другой нет) — оранжевый «жанр-чувствительно»: на
# гуманитарных/академических текстах v4 завышает (стиль близок к обучающим), v3 нет.
AGREE_AI = "оба: ИИ"
AGREE_HUMAN = "оба: человек"
AGREE_DISAGREE = "расхождение (жанр)"
AGREE_INSUFF = "недостаточно текста"
AGREE_STYLE = {
    AGREE_AI:       ("v-ai",        "red",        "ОБА (v3+v4): ИИ — уверенно"),
    AGREE_HUMAN:    ("v-human",     "green",      "ОБА (v3+v4): человек"),
    AGREE_DISAGREE: ("v-disagree",  "dark_orange","РАСХОЖДЕНИЕ — жанр-чувствительно"),
    AGREE_INSUFF:   ("v-insufficient", "bright_black", "НЕДОСТАТОЧНО ТЕКСТА"),
}


@dataclass
class Report:
    path: str
    model_key: str
    threshold: float           # порог ПОДСВЕТКИ карты (display-only, мягкий)
    n_total: int
    paragraphs: list           # list[ds.ParaResult]  (idx, text, n_chars, p_ai, label)
    dropped: list              # list[(ds.Paragraph, reason)]
    verdict_threshold: float = 0.0   # боевой FPR-порог модели — ТОЛЬКО для показа
    warnings: list = field(default_factory=list)
    error: Optional[str] = None
    glued: list = field(default_factory=list)   # list[ds.GluedFragment] — диагностика 'g'
    glued_aggr: list = field(default_factory=list)  # агрессивная склейка — диагностика 'G'
    genre: Optional[dict] = None                 # вердикт genre_classify.classify (жанр+рекоменд.)


_V1 = None   # (clf, thr) — грузим один раз


def _scan_v1(text: str, blocks=None):
    """
    v1: тот же путь, что у сервиса — features + model.joblib (document_scan).
    Возвращает (dict, verdict_thr). КАРТА красится по мягкому HIGHLIGHT_THR["v1"];
    боевой вердиктный порог (0.86) отдаём отдельно, ТОЛЬКО для показа.
    """
    global _V1
    if _V1 is None:
        _V1 = ds.load_model()           # model.joblib + threshold_v1.json (0.86)
    clf, verdict_thr = _V1
    # scan_document(...thr) использует thr ровно для разметки абзацев (labels) —
    # подаём сюда МЯГКИЙ порог карты, а не вердиктный. Модель/окна не меняются.
    res = ds.scan_document(text, clf, HIGHLIGHT_THR["v1"], blocks=blocks)
    # диагностика: склейка соседней короткой прозы -> скор тем же v1
    v1_score = lambda ts: ds.predict_v1(clf, ts)
    glued = ds.glue_short_prose(res["paragraphs"])
    ds.score_fragments(glued, v1_score, HIGHLIGHT_THR["v1"])
    res["glued"] = glued
    table_idxs = {p.idx for p, r in res["dropped"] if "таблиц" in r}
    glued_aggr = ds.glue_short_prose_aggressive(res["paragraphs"], table_idxs)
    ds.score_fragments(glued_aggr, v1_score, HIGHLIGHT_THR["v1"])
    res["glued_aggr"] = glued_aggr
    return res, verdict_thr


def _scan_transformer(text: str, version: str, blocks=None):
    """
    v2/v3/v4: тот же конвейер document_scan, но окно скорится ТРАНСФОРМЕРОМ версии.
    Переиспользуем функции document_scan; меняется только источник p(ИИ) окна.
    Разметка абзацев — по мягкому HIGHLIGHT_THR[version]; вердиктный порог отдаём
    отдельно, ТОЛЬКО для показа. Окна длинные (>=150 слов), length-aware не нужен.
    """
    from aidetector import transformer_score as TF      # ленивый импорт: torch грузится только для трансформера
    sc = TF.get(version)
    verdict_thr = sc.threshold()        # базовый вердиктный порог версии (НЕ для карты)
    paras = ds.split_paragraphs(text)
    ds.attach_block_meta(paras, blocks)             # XML-мета для подкатегорий коротких
    split = ds.classify_sections(paras)
    stream, spans = ds._build_content_stream(split.content)
    wins = ds.build_windows(stream)
    if wins:
        probs = sc.proba([stream[w.cstart:w.cend] for w in wins])   # батч, кэш по md5
        for w, p in zip(wins, probs):
            w.p_ai = float(p)
    # разметка по МЯГКОМУ порогу карты (не по вердиктному)
    para_results = ds.project_to_paragraphs(split.content, spans, wins, HIGHLIGHT_THR[version])
    # диагностика: склейка соседней короткой прозы -> скор тем же трансформером
    glued = ds.glue_short_prose(para_results)
    ds.score_fragments(glued, sc.proba, HIGHLIGHT_THR[version])
    table_idxs = {p.idx for p, r in split.dropped if "таблиц" in r}
    glued_aggr = ds.glue_short_prose_aggressive(para_results, table_idxs)
    ds.score_fragments(glued_aggr, sc.proba, HIGHLIGHT_THR[version])
    return ({"threshold": HIGHLIGHT_THR[version], "n_paragraphs_total": len(paras),
             "dropped": split.dropped, "windows": wins, "paragraphs": para_results,
             "glued": glued, "glued_aggr": glued_aggr}, verdict_thr)


def run_scan(path, model_key: str, genre: Optional[dict] = None) -> Report:
    """Файл + модель -> Report. Никаких исключений наружу — ошибки в Report.error.
    genre — вердикт автоклассификатора жанра (для показа), модель уже выбрана выше."""
    path = Path(path)
    text, warnings, blocks = extract_blocks(path)
    if not text.strip():
        return Report(path=str(path), model_key=model_key, threshold=0.0, n_total=0,
                      paragraphs=[], dropped=[], warnings=warnings, genre=genre,
                      error="Текста для анализа нет (см. предупреждение). "
                            "Сканы требуют OCR — это отдельный этап, пока не реализован.")
    res, verdict_thr = (_scan_v1(text, blocks) if model_key == "v1"
                        else _scan_transformer(text, model_key, blocks))
    return Report(path=str(path), model_key=model_key, threshold=res["threshold"],
                  verdict_threshold=verdict_thr, n_total=res["n_paragraphs_total"],
                  paragraphs=res["paragraphs"], dropped=res["dropped"], warnings=warnings,
                  glued=res.get("glued", []), glued_aggr=res.get("glued_aggr", []), genre=genre)


def summarize(report: Report):
    """Доли по категориям (по символам проанализированного текста)."""
    total = sum(r.n_chars for r in report.paragraphs) or 1
    rows = []
    for cat in CATS:
        chars = sum(r.n_chars for r in report.paragraphs if r.label == cat)
        n = sum(1 for r in report.paragraphs if r.label == cat)
        rows.append((cat, n, chars, chars / total))
    return rows


def insufficient_breakdown(report: Report):
    """Разбивка 'недостаточно текста' по подкатегориям (структура vs короткая проза).
    Возвращает список (подкатегория, кол-во) по убыванию. Короткая проза вынесена
    наверх — это потенциальная лазейка (ИИ, нарезанный на мелкие тезисы)."""
    from collections import Counter
    c = Counter(r.subcat or "—" for r in report.paragraphs
                if r.label == "недостаточно текста" and r.p_ai is None)
    order = {ds.SUBCAT_PROSE: 0}        # короткая проза — первой
    return sorted(c.items(), key=lambda kv: (order.get(kv[0], 1), -kv[1]))


def glue_stats(report: Report, frags=None) -> dict:
    """Статистика склейки для сводки/экспорта. frags=None → обычная склейка
    (report.glued); можно передать report.glued_aggr для агрессивной. Доля подозрительного
    «с учётом склейки» — ПО СИМВОЛАМ (как основная сводка): символы склеенных подозрительных
    фрагментов = символы их абзацев-членов (они уже в знаменателе как «недостаточно текста»),
    знаменатель — все символы контента, не меняется."""
    if frags is None:
        frags = report.glued
    total = sum(r.n_chars for r in report.paragraphs) or 1
    base_susp = sum(r.n_chars for r in report.paragraphs if r.label == "подозрительно")
    idx2ch = {r.idx: r.n_chars for r in report.paragraphs}
    susp = uncert = human = 0
    susp_chars = frag_chars = 0
    for f in frags:
        mc = sum(idx2ch.get(i, 0) for i in f.member_idxs)
        frag_chars += mc
        if f.label == "подозрительно":
            susp += 1
            susp_chars += mc
        elif f.label == "неопределённо":
            uncert += 1
        else:
            human += 1
    return {"n": len(frags), "susp": susp, "uncert": uncert, "human": human,
            "frag_chars": frag_chars, "base_share": base_susp / total,
            "with_share": (base_susp + susp_chars) / total}


@dataclass
class AgreeResult:
    idx: int
    text: str
    n_chars: int
    p_v3: Optional[float]
    p_v4: Optional[float]
    label: str


def agreement_results(path):
    """Скорит документ И v3, И v4, проецирует на абзацы, размечает согласованность.
    Сравнивает по порогам КАРТЫ каждой версии (v3=0.60, v4=0.72): оба>=порога → ИИ;
    оба<порога → человек; иначе → расхождение (жанр-чувствительно). Возвращает
    (list[AgreeResult], dropped, warnings). Модели/пороги не трогает — только читает."""
    text, warnings, blocks = extract_blocks(path)
    if not text.strip():
        return [], [], warnings
    paras = ds.split_paragraphs(text)
    ds.attach_block_meta(paras, blocks)
    split = ds.classify_sections(paras)
    stream, spans = ds._build_content_stream(split.content)
    wins = ds.build_windows(stream)
    from aidetector import transformer_score as TF
    by_ver = {}
    for ver in ("v3", "v4"):
        sc = TF.get(ver)
        if wins:
            for w, p in zip(wins, sc.proba([stream[w.cstart:w.cend] for w in wins])):
                w.p_ai = float(p)
        by_ver[ver] = {r.idx: r for r in
                       ds.project_to_paragraphs(split.content, spans, wins, HIGHLIGHT_THR[ver])}
    out = []
    for idx, r4 in by_ver["v4"].items():
        r3 = by_ver["v3"].get(idx)
        p4, p3 = r4.p_ai, (r3.p_ai if r3 else None)
        if p3 is None or p4 is None:
            lab = AGREE_INSUFF
        else:
            f3, f4 = p3 >= HIGHLIGHT_THR["v3"], p4 >= HIGHLIGHT_THR["v4"]
            lab = AGREE_AI if (f3 and f4) else AGREE_HUMAN if (not f3 and not f4) else AGREE_DISAGREE
        out.append(AgreeResult(idx, r4.text, r4.n_chars, p3, p4, lab))
    return out, split.dropped, warnings


def agreement_counts(agreement) -> dict:
    from collections import Counter
    c = Counter(a.label for a in agreement)
    return {k: c.get(k, 0) for k in (AGREE_AI, AGREE_HUMAN, AGREE_DISAGREE, AGREE_INSUFF)}


def quick_model_agreement(text: str, sample: int = 30) -> dict:
    """БЫСТРАЯ проверка согласия v3/v4: равномерная выборка до `sample` окон по всему
    документу (не весь док — экономия), скор обеими моделями. Метрика — доля окон с
    СИЛЬНЫМ расхождением: v4 видит ИИ (>=порога карты), а v3 уверенно человек (<0.45).
    Именно это ловит ЖАНРОВЫЙ ПЕРЕКОС v4 (а не «v3 чуть недотянул» — там v3 тоже ~0.5).
    -> {assessable, div, n}."""
    paras = ds.split_paragraphs(text)
    split = ds.classify_sections(paras)
    stream, spans = ds._build_content_stream(split.content)
    wins = ds.build_windows(stream)
    if not wins:
        return {"assessable": False, "div": 0.0, "n": 0}
    if len(wins) > sample:
        step = len(wins) / sample
        wins = [wins[int(i * step)] for i in range(sample)]
    texts = [stream[w.cstart:w.cend] for w in wins]
    from aidetector import transformer_score as TF
    p3 = TF.get("v3").proba(texts)
    p4 = TF.get("v4").proba(texts)
    if len(wins) < 3:
        return {"assessable": False, "div": 0.0, "n": len(wins)}
    strong = sum(1 for a, b in zip(p3, p4)
                 if b >= HIGHLIGHT_THR["v4"] and a < AGREE_V3_HUMAN)
    return {"assessable": True, "div": strong / len(wins), "n": len(wins)}


def recommend(text: str) -> dict:
    """Финальная рекомендация = жанр (genre_classify) + второй слой согласия моделей.
    Уверенный технический жанр с сильным расхождением v3/v4 → автосогласованность
    (закрывает кейс Антик1: жанр технич., но v4 перекошен). Гуманитарный → v3 остаётся
    (там v3 и есть безопасный выбор). Возвращает обогащённый genre-вердикт."""
    g = genre_classify.classify(text)
    g["agreement"] = None
    g["overridden"] = False
    if g.get("confident"):
        agr = quick_model_agreement(text)
        g["agreement"] = agr
        if (agr["assessable"] and agr["div"] > AGREE_DIV_THRESH
                and g["recommended"] == "v4"):
            g["recommended"] = "agree"
            g["overridden"] = True
            g["reason"] = (f"жанр {g['genre']}, НО v3/v4 сильно расходятся "
                           f"({agr['div']:.0%}: v4 видит ИИ там, где v3 уверенно человек) → "
                           "режим согласованности, не доверять v4 слепо")
    return g


# =============================================================================
# HEADLESS текстовый рендер (ANSI) — для --text, проверки и CI без терминала-TUI
# =============================================================================
class _A:
    R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GRN = "\033[32m"; YEL = "\033[33m"; GREY = "\033[90m"


_ANSI = {"подозрительно": _A.RED, "неопределённо": _A.YEL,
         "человек": _A.GRN, "недостаточно текста": _A.GREY}


def _prev(text: str, n: int = 100) -> str:
    import re
    one = re.sub(r"\s+", " ", text).strip()
    return one if len(one) <= n else one[:n - 1] + "…"


def render_ansi(report: Report, model_key: str, glue_on: bool = False) -> str:
    out = [f"\n{_A.B}Документ:{_A.R} {Path(report.path).name}  "
           f"({report.n_total} абзацев всего)   "
           f"{_A.B}модель:{_A.R} {MODEL_SHORT[model_key]}  "
           f"порог карты {report.threshold:.2f} {_A.DIM}(вердиктный {report.verdict_threshold:.2f}, не для карты){_A.R}"]
    for w in report.warnings:
        out.append(f"{_A.YEL}[!] {w}{_A.R}")
    if report.error:
        out.append(f"{_A.RED}{report.error}{_A.R}")
        return "\n".join(out)

    # карта абзацев в исходном порядке (контент + служебное вперемешку по idx)
    items = [(r.idx, "c", r) for r in report.paragraphs]
    items += [(p.idx, "s", (p, reason)) for p, reason in report.dropped]
    items.sort(key=lambda x: x[0])
    out.append(f"\n{_A.B}=== КАРТА АБЗАЦЕВ ==={_A.R}")
    for idx, knd, payload in items:
        if knd == "c":
            r = payload
            c = _ANSI.get(r.label, _A.R)
            p = "  —  " if r.p_ai is None else f"p(ИИ)={r.p_ai:.3f}"
            _, _, name = STYLE[r.label]
            out.append(f"  {c}●{_A.R} абз.{idx:>2} [{c}{name}{_A.R}] {p} {_A.DIM}({r.n_chars} симв){_A.R}")
            out.append(f"      {_A.DIM}{_prev(r.text)}{_A.R}")
        else:
            p, reason = payload
            out.append(f"  {_A.GREY}○ абз.{idx:>2} [СЛУЖЕБНОЕ — не анализировалось] {reason}{_A.R}")
            out.append(f"      {_A.GREY}{_prev(p.text)}{_A.R}")

    # сводка
    out.append(f"\n{_A.B}=== СВОДКА (доля проанализированного текста) ==={_A.R}")
    for cat, n, chars, share in summarize(report):
        if chars == 0:
            continue
        c = _ANSI.get(cat, _A.R)
        bar = "█" * int(round(share * 24))
        out.append(f"  {c}{cat:<20}{_A.R} {share:5.1%}  {c}{bar}{_A.R}  ({n} абз., {chars} симв)")
    if report.dropped:
        out.append(f"  {_A.GREY}служебное отсеяно: {len(report.dropped)} абз.{_A.R}")
    bd = insufficient_breakdown(report)
    if bd:
        parts = "  ".join(f"{s}: {n}" for s, n in bd)
        out.append(f"  {_A.DIM}из «недостаточно текста»: {parts}{_A.R}")
        n_prose = next((n for s, n in bd if s == ds.SUBCAT_PROSE), 0)
        if n_prose and report.glued and not glue_on:
            out.append(f"  {_A.DIM}короткой прозы {n_prose} — склейка соседних даёт "
                       f"{len(report.glued)} фрагм. (флаг --glue покажет вердикты){_A.R}")
    if glue_on:
        g = glue_stats(report)
        out.append(f"\n  {_A.B}=== С УЧЁТОМ СКЛЕЙКИ ПРОЗЫ (диагностика) ==={_A.R}")
        if g["n"] == 0:
            out.append(f"  {_A.DIM}соседней короткой прозы для склейки нет → +0{_A.R}")
        else:
            out.append(f"  при склейке: {_A.RED}+{g['susp']} подозрительно{_A.R}, "
                       f"{_A.YEL}+{g['uncert']} неопределённо{_A.R} "
                       f"{_A.DIM}(из {g['n']} фрагм., {g['frag_chars']} симв){_A.R}")
            out.append(f"  {_A.B}общая доля подозрительного с учётом склейки: "
                       f"{g['with_share']:.1%}{_A.R} {_A.DIM}(без склейки {g['base_share']:.1%}){_A.R}")
    out.append(f"\n  {_A.YEL}Вероятностная оценка по фрагментам, не доказательство авторства.{_A.R}")
    out.append(f"  {_A.DIM}Недопустимо как основание для дисциплинарных/академических решений.{_A.R}")
    return "\n".join(out)


# =============================================================================
# HEADLESS: согласованность v3+v4 (для --agree / проверки)
# =============================================================================
def render_agreement_ansi(path) -> str:
    agreement, dropped, warnings = agreement_results(path)
    out = [f"\n{_A.B}=== СОГЛАСОВАННОСТЬ v3 + v4 — {Path(path).name} ==={_A.R}"]
    for w in warnings:
        out.append(f"{_A.YEL}[!] {w}{_A.R}")
    if not agreement:
        out.append("  нет абзацев с вердиктом")
        return "\n".join(out)
    cnt = agreement_counts(agreement)
    verd = sum(v for k, v in cnt.items() if k != AGREE_INSUFF) or 1
    out.append(f"  {_A.RED}оба:ИИ {cnt[AGREE_AI]}{_A.R}   {_A.GRN}оба:человек {cnt[AGREE_HUMAN]}{_A.R}   "
               f"расхождение(жанр) {cnt[AGREE_DISAGREE]}   {_A.DIM}недостаточно {cnt[AGREE_INSUFF]}{_A.R}")
    out.append(f"  доля расхождений среди размеченных: {cnt[AGREE_DISAGREE]/verd:.0%}")
    dis = [a for a in agreement if a.label == AGREE_DISAGREE]
    if dis:
        out.append("\n  топ расхождений (v3 видит человека, v4 — ИИ):")
        for a in sorted(dis, key=lambda a: -(abs((a.p_v4 or 0) - (a.p_v3 or 0))))[:8]:
            out.append(f"   абз.{a.idx:>3} v3={a.p_v3:.2f} v4={a.p_v4:.2f} "
                       f"Δ={abs(a.p_v4-a.p_v3):.2f}  {_prev(a.text, 66)}")
    out.append(f"\n  {_A.DIM}расхождение = один видит ИИ, другой нет; на гуманитарных/академических "
               f"работах v4 склонен ЗАВЫШАТЬ (стиль близок к обучающим). Перепроверять.{_A.R}")
    return "\n".join(out)


# =============================================================================
# ЭКСПОРТ ОТЧЁТА (markdown + HTML) — UI-фича, модель/сегментацию не трогает
# =============================================================================
_EXPORT_DISCLAIMER = (
    "Вероятностная оценка по фрагментам — НЕ доказательство авторства. Карта (мягкий "
    "порог) — зона внимания, не обвинение. Шаблонные места (схемы БД, формулы, "
    "канцелярит введения) подсвечиваются у любых авторов — это особенность жанра, не "
    "обязательно ИИ. ОГРАНИЧЕНИЕ v4: на гуманитарных/лингвистических работах с плотным "
    "академическим стилем v4 может завышать оценку (стилистическая близость к текстам "
    "обучения) — рекомендуется проверка через v3 для согласованности. Недопустимо как "
    "основание для дисциплинарных/академических решений."
)
# приглушённые фоны абзацев для HTML (как в TUI: красный/жёлтый/зелёный/серый)
_HTML_BG = {
    "подозрительно":       ("#f8d2d6", "#c0392b"),
    "неопределённо":       ("#fdf3cf", "#b9770e"),
    "человек":             ("#d6efd8", "#27733a"),
    "недостаточно текста": ("#eceef0", "#9aa0a6"),
}
_MD_TAG = {"подозрительно": "🔴 ПОДОЗРИТЕЛЬНО", "неопределённо": "🟡 НЕОПРЕДЕЛЁННО",
           "человек": "🟢 ЧЕЛОВЕК", "недостаточно текста": "⚪ НЕДОСТАТОЧНО ТЕКСТА"}


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _report_header_rows(report: Report):
    return [
        ("Файл", Path(report.path).name),
        ("Дата проверки", _now()),
        ("Модель", MODEL_FULL.get(report.model_key, report.model_key)),
        ("Порог карты (подсветка)", f"{report.threshold:.2f}"),
        ("Вердиктный порог модели", f"{report.verdict_threshold:.2f} (не для карты)"),
        ("Абзацев всего / служебных отсеяно", f"{report.n_total} / {len(report.dropped)}"),
    ]


def report_markdown(report: Report, glue_on: bool = False) -> str:
    L = [f"# Отчёт проверки на ИИ — {Path(report.path).name}", ""]
    for k, v in _report_header_rows(report):
        L.append(f"- **{k}:** {v}")
    L.append("\n## Сводка")
    for cat, n, chars, share in summarize(report):
        if chars == 0 and n == 0:
            continue
        L.append(f"- {_MD_TAG.get(cat, cat)}: **{share:.1%}** ({n} абз., {chars} симв)")
    bd = insufficient_breakdown(report)
    if bd:
        L.append("- из «недостаточно текста»: " + ", ".join(f"{s} — {n}" for s, n in bd))
    if glue_on and report.glued:
        g = glue_stats(report)
        L.append(f"- **с учётом склейки прозы:** +{g['susp']} подозрительно, "
                 f"+{g['uncert']} неопределённо (из {g['n']} фрагм.); общая доля "
                 f"подозрительного {g['with_share']:.1%} (без склейки {g['base_share']:.1%})")
    L.append("\n## Абзацы с вердиктом")
    verd = [r for r in report.paragraphs if r.p_ai is not None]
    for r in sorted(verd, key=lambda r: r.idx):
        L.append(f"\n**абз. {r.idx}** — {_MD_TAG.get(r.label, r.label)} · "
                 f"p(ИИ)={r.p_ai:.3f} · {r.n_chars} симв\n\n> "
                 + r.text.replace("\n", " ").strip())
    if glue_on and report.glued:           # раздел только если склейка была включена
        L.append("\n## Диагностика: склейка коротких прозаических")
        L.append("_Соседние короткие прозаические абзацы (не разделённые заголовком/"
                 "таблицей/подписью) склеены до ≥200 симв и проверены — показывают ИИ, "
                 "скрытый мелкой нарезкой на тезисы. Документ при этом не менялся._")
        g = glue_stats(report)
        L.append(f"\n**Итог склейки:** +{g['susp']} подозрительно, +{g['uncert']} неопределённо "
                 f"из {g['n']} фрагм.; общая доля подозрительного с учётом склейки "
                 f"**{g['with_share']:.1%}** (без склейки {g['base_share']:.1%}).")
        for f in sorted(report.glued, key=lambda f: -(f.p_ai or 0)):
            L.append(f"\n**склейка абз. {f.member_idxs[0]}–{f.member_idxs[-1]}** "
                     f"({len(f.member_idxs)} абз.) — {_MD_TAG.get(f.label, f.label)} · "
                     f"p(ИИ)={f.p_ai:.3f}\n\n> " + f.text.replace("\n", " ").strip())
    dropped = report.dropped
    if dropped:
        L.append("\n## Отсеянные служебные секции")
        from collections import Counter
        for reason, n in Counter(r for _, r in dropped).most_common():
            L.append(f"- {reason}: {n} абз.")
    L.append("\n---\n")
    L.append("> ⚠ " + _EXPORT_DISCLAIMER)
    return "\n".join(L)


def report_html(report: Report, glue_on: bool = False) -> str:
    import html as _h
    esc = _h.escape

    def para_block(idx, label, p_str, n_chars, text, prefix=""):
        bg, bar = _HTML_BG.get(label, _HTML_BG["недостаточно текста"])
        return (f'<div class="para" id="p{idx}" '
                f'style="background:{bg};border-left:6px solid {bar};">'
                f'<div class="meta">{prefix}абз. {esc(str(idx))} — {esc(label.upper())} '
                f'· p(ИИ)={p_str} · {n_chars} симв · '
                f'<a href="#p{idx}">#</a></div>'
                f'<div class="txt">{esc(text.replace(chr(10), " ").strip())}</div></div>')

    rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>"
                   for k, v in _report_header_rows(report))
    summ = ""
    for cat, n, chars, share in summarize(report):
        if n == 0 and chars == 0:
            continue
        bg, bar = _HTML_BG.get(cat, ("#eee", "#999"))
        summ += (f'<li><span class="chip" style="background:{bg};border-color:{bar}">'
                 f'{esc(cat)}</span> <b>{share:.1%}</b> ({n} абз., {chars} симв)</li>')
    bd = insufficient_breakdown(report)
    bd_html = ("<p class='dim'>из «недостаточно текста»: "
               + ", ".join(f"{esc(s)} — {n}" for s, n in bd) + "</p>") if bd else ""

    verd = sorted([r for r in report.paragraphs if r.p_ai is not None], key=lambda r: r.idx)
    paras_html = "".join(para_block(r.idx, r.label, f"{r.p_ai:.3f}", r.n_chars, r.text)
                         for r in verd)

    glued_html = ""
    if glue_on and report.glued:           # раздел только если склейка была включена
        g = glue_stats(report)
        glued_html = ("<h2>Диагностика: склейка коротких прозаических</h2>"
                      "<p class='dim'>Соседние короткие прозаические абзацы (не разделённые "
                      "заголовком/таблицей/подписью) склеены до ≥200 симв и проверены — "
                      "показывают ИИ, скрытый мелкой нарезкой. Документ не менялся.</p>"
                      f"<p><b>Итог склейки:</b> +{g['susp']} подозрительно, +{g['uncert']} "
                      f"неопределённо из {g['n']} фрагм.; общая доля подозрительного с учётом "
                      f"склейки <b>{g['with_share']:.1%}</b> (без склейки {g['base_share']:.1%}).</p>")
        glued_html += "".join(
            para_block(f"g{f.member_idxs[0]}", f.label, f"{f.p_ai:.3f}", f.n_chars, f.text,
                       prefix=f"склейка {len(f.member_idxs)} абз. ({f.member_idxs[0]}–{f.member_idxs[-1]}) · ")
            for f in sorted(report.glued, key=lambda f: -(f.p_ai or 0)))

    from collections import Counter
    drop_html = ""
    if report.dropped:
        drop_html = "<h2>Отсеянные служебные секции</h2><ul>" + "".join(
            f"<li>{esc(reason)}: {n} абз.</li>"
            for reason, n in Counter(r for _, r in report.dropped).most_common()) + "</ul>"

    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Отчёт — {esc(Path(report.path).name)}</title>
<style>
  body {{ font-family: "Times New Roman", Georgia, serif; max-width: 900px;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.25rem; margin-top: 1.8rem;
         border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  table.hdr td {{ padding: 2px 10px 2px 0; vertical-align: top; }}
  table.hdr td:first-child {{ color: #555; white-space: nowrap; }}
  ul.summary {{ list-style: none; padding: 0; }}
  ul.summary li {{ margin: .25rem 0; }}
  .chip {{ display: inline-block; padding: 1px 8px; border-radius: 10px;
           border: 1px solid; font-size: .85em; }}
  .para {{ margin: .6rem 0; padding: .5rem .8rem; border-radius: 4px; }}
  .para .meta {{ font-size: .8em; color: #444; margin-bottom: .25rem; }}
  .para .meta a {{ text-decoration: none; color: #888; }}
  .para .txt {{ white-space: pre-wrap; }}
  .dim {{ color: #777; font-size: .9em; }}
  .disclaimer {{ margin-top: 2rem; padding: 1rem; background: #fff8e1;
                 border-left: 6px solid #f0ad4e; font-size: .95em; }}
</style></head><body>
<h1>Отчёт проверки на ИИ — {esc(Path(report.path).name)}</h1>
<table class="hdr">{rows}</table>
<h2>Сводка</h2><ul class="summary">{summ}</ul>{bd_html}
<h2>Абзацы с вердиктом</h2>{paras_html}
{glued_html}
{drop_html}
<div class="disclaimer">⚠ {esc(_EXPORT_DISCLAIMER)}</div>
</body></html>"""


def export_report_files(report: Report, out_dir=None, glue_on: bool = False) -> list:
    """Сохраняет .report.md и .report.html рядом с документом (или в out_dir).
    glue_on=True — добавляет раздел «склейка коротких прозаических» (как было на экране
    в момент экспорта); False — раздела нет. Возвращает список путей.
    UI-фича — модель/пороги/сегментацию не трогает."""
    src = Path(report.path)
    out = Path(out_dir) if out_dir else src.parent
    # имя строим явно (with_suffix съел бы «.report» как расширение)
    md_path = out / f"{src.stem}.report.md"
    html_path = out / f"{src.stem}.report.html"
    md_path.write_text(report_markdown(report, glue_on=glue_on), encoding="utf-8")
    html_path.write_text(report_html(report, glue_on=glue_on), encoding="utf-8")
    return [md_path, html_path]


# =============================================================================
# TUI (Textual)
# =============================================================================
from rich.markup import escape                                     # noqa: E402
from textual import work                                           # noqa: E402
from textual.app import App, ComposeResult                         # noqa: E402
from textual.containers import Vertical, VerticalScroll            # noqa: E402
from textual.screen import Screen                                  # noqa: E402
from textual.widgets import (DirectoryTree, Footer, Header, Input,  # noqa: E402
                             LoadingIndicator, Static)

DISCLAIMER = ("[yellow]⚠ Вероятностная оценка по фрагментам — НЕ доказательство авторства.[/] "
              "[dim]Карта (мягкий порог 0.5–0.6) — ЗОНА ВНИМАНИЯ, не обвинение: она плавает "
              "на композиции окна (переоформление двигает её на ±0.05–0.24). Жёсткий вердикт "
              "≥0.846 устойчив к оформлению (медиана Δp по идент. тексту 0.02), но и он не "
              "доказательство. Шаблонные места (схемы БД, формулы, канцелярит введения) "
              "подсвечиваются у всех — это особенность жанра, не обязательно ИИ.[/] "
              "[dark_orange]Ограничение v4: на гуманитарных/лингвистических работах с плотным "
              "академическим стилем v4 может ЗАВЫШАТЬ оценку (стиль близок к обучающим текстам). "
              "Сверяйтесь с v3 — клавиша [b]a[/] (согласованность): оранжевые абзацы = модели "
              "разошлись, перепроверить.[/] [dim]Недопустимо как основание для "
              "дисциплинарных/академических решений.[/]")


def _para_static(r) -> Static:
    """Блок одного содержательного абзаца, подсвеченный по вердикту."""
    cls, color, name = STYLE[r.label]
    p = "—" if r.p_ai is None else f"{r.p_ai:.3f}"
    head = f"[b {color}]● абз.{r.idx}: {name}[/]  [dim]p(ИИ)={p} · {r.n_chars} симв[/]"
    return Static(f"{head}\n{escape(r.text)}", classes=f"para {cls}")


def _service_static(p, reason) -> Static:
    """Отсеянная служебная секция — приглушённо, помечена, текст НЕ теряется."""
    head = f"[b dim]○ абз.{p.idx}: СЛУЖЕБНОЕ — не анализировалось[/]  [dim]({reason})[/]"
    return Static(f"{head}\n{escape(p.text)}", classes="para v-service")


def _glued_static(frag) -> Static:
    """Диагностический склееный фрагмент (соседняя короткая проза) — как обычный
    абзац карты, но с пометкой «(склейка N абзацев)»."""
    label = frag.label or "недостаточно текста"
    cls, color, name = STYLE.get(label, STYLE["недостаточно текста"])
    p = "—" if frag.p_ai is None else f"{frag.p_ai:.3f}"
    head = (f"[b {color}]⊕ склейка {len(frag.member_idxs)} абз. "
            f"(абз.{frag.member_idxs[0]}–{frag.member_idxs[-1]}): {name}[/]  "
            f"[dim]p(ИИ)={p} · {frag.n_chars} симв · диагностика[/]")
    return Static(f"{head}\n{escape(frag.text)}", classes=f"para {cls} glued")


def _aggr_static(frag) -> Static:
    """Фрагмент АГРЕССИВНОЙ склейки — фиолетовая дашед-рамка, метка «⊗ агр. склейка»."""
    label = frag.label or "недостаточно текста"
    cls, color, name = STYLE.get(label, STYLE["недостаточно текста"])
    p = "—" if frag.p_ai is None else f"{frag.p_ai:.3f}"
    head = (f"[b magenta]⊗ агр. склейка {len(frag.member_idxs)} абз. "
            f"(абз.{frag.member_idxs[0]}–{frag.member_idxs[-1]}):[/] [b {color}]{name}[/]  "
            f"[dim]p(ИИ)={p} · {frag.n_chars} симв · через структ. границы[/]")
    return Static(f"{head}\n{escape(frag.text)}", classes=f"para {cls} glued-aggr")


def _agree_static(a) -> Static:
    """Абзац в режиме согласованности: цвет по согласию v3/v4, оба p показаны."""
    cls, color, name = AGREE_STYLE.get(a.label, AGREE_STYLE[AGREE_INSUFF])
    p3 = "—" if a.p_v3 is None else f"{a.p_v3:.2f}"
    p4 = "—" if a.p_v4 is None else f"{a.p_v4:.2f}"
    head = f"[b {color}]◧ абз.{a.idx}: {name}[/]  [dim]v3={p3} · v4={p4} · {a.n_chars} симв[/]"
    return Static(f"{head}\n{escape(a.text)}", classes=f"para {cls}")


def _agreement_summary_markup(report: Report, agreement) -> str:
    cnt = agreement_counts(agreement)
    verd = sum(v for k, v in cnt.items() if k != AGREE_INSUFF) or 1
    gl = _genre_line(report)
    return "\n".join(([gl] if gl else []) + [
        f"[b]РЕЖИМ СОГЛАСОВАННОСТИ v3 + v4[/]  [dim](по абзацам, где у обеих есть вердикт)[/]",
        f"[red]оба: ИИ {cnt[AGREE_AI]}[/]   ·   [green]оба: человек {cnt[AGREE_HUMAN]}[/]   ·   "
        f"[dark_orange]расхождение (жанр) {cnt[AGREE_DISAGREE]}[/]   ·   "
        f"[dim]недостаточно {cnt[AGREE_INSUFF]}[/]",
        f"[dim]расхождений среди размеченных: {cnt[AGREE_DISAGREE]/verd:.0%}[/]",
        "[dark_orange]Оранжевые = v3 и v4 не сошлись. На гуманитарных/лингвистических "
        "академических работах v4 склонен ЗАВЫШАТЬ (стиль близок к обучающим текстам) — "
        "читайте такие абзацы как «перепроверить через v3», а не «ИИ».[/]  "
        "[dim](a — выкл)[/]",
    ])


def _genre_line(report: Report) -> str:
    """Строка-рекомендация жанра/модели для шапки сводки (если жанр определён)."""
    g = report.genre
    if not g:
        return ""
    recname = {"v3": "v3", "v4": "v4", "agree": "режим согласованности (a)"}[g["recommended"]]
    mark = "dark_orange" if (g.get("overridden") or not g["confident"]) else "green"
    line = (f"[{mark}]Жанр:[/] {g['genre']} · [b]рекомендую {recname}[/] "
            f"[dim]({g['reason']})[/]")
    agr = g.get("agreement")
    if agr and agr.get("assessable"):
        lvl = "низкое" if agr["div"] > AGREE_DIV_THRESH else "высокое"
        clr = "dark_orange" if agr["div"] > AGREE_DIV_THRESH else "green"
        line += (f"\n[{clr}]согласие моделей: {lvl}[/] [dim]({agr['div']:.0%} окон — "
                 f"v4 видит ИИ там, где v3 уверенно человек; из {agr['n']} проверенных)[/]")
    return line


def _summary_markup(report: Report, glue_on: bool = False, aggr_on: bool = False) -> str:
    lines = []
    gl = _genre_line(report)
    if gl:
        lines.append(gl)
    lines += [f"[b]Модель:[/] {MODEL_FULL[report.model_key]}",
             f"[b]Порог подсветки карты:[/] {report.threshold:.2f}  "
             f"[dim](мягкий, только для цвета; вердиктный FPR-порог модели {report.verdict_threshold:.2f} "
             f"не меняется и для карты не применяется)[/]",
             f"[dim]абзацев всего: {report.n_total}, служебных отсеяно: {len(report.dropped)}[/]"]
    for cat, n, chars, share in summarize(report):
        _, color, _ = STYLE[cat]
        bar = "█" * int(round(share * 22))
        lines.append(f"[{color}]{cat:<20}[/] {share:5.1%} [{color}]{bar}[/] [dim]({n} абз., {chars} симв)[/]")
    # подкатегории «недостаточно текста»: ЧТО именно не попало в анализ (XML+эвристика)
    bd = insufficient_breakdown(report)
    if bd:
        parts = []
        for sub, n in bd:
            tag = f"[b yellow]{sub} {n}[/]" if sub == ds.SUBCAT_PROSE else f"{sub} {n}"
            parts.append(tag)
        lines.append(f"[dim]   └ из «недостаточно»:[/] " + " · ".join(parts))
        n_prose = next((n for s, n in bd if s == ds.SUBCAT_PROSE), 0)
        if n_prose and report.glued and not glue_on:
            lines.append(f"[dim]   короткой прозы {n_prose} — клавиша [b]g[/] склеит соседние "
                         f"и проверит ({len(report.glued)} фрагм.)[/]")
    # ВТОРОЙ блок — только при включённой склейке (g). Основную сводку НЕ подменяет.
    if glue_on:
        g = glue_stats(report)
        lines.append("[b]— с учётом склейки прозы (диагностика, документ не меняется):[/]")
        if g["n"] == 0:
            lines.append("[dim]   соседней короткой прозы для склейки нет → +0[/]")
        else:
            lines.append(
                f"   при склейке коротких прозаических: [red]+{g['susp']} подозрительно[/], "
                f"[yellow]+{g['uncert']} неопределённо[/] "
                f"[dim](из {g['n']} склеенных фрагм., {g['frag_chars']} симв)[/]")
            lines.append(
                f"   [b]общая доля подозрительного с учётом склейки: {g['with_share']:.1%}[/] "
                f"[dim](по символам; без склейки было {g['base_share']:.1%})[/]")
    # ТРЕТИЙ блок — агрессивная склейка (через структурные границы). Только при aggr_on.
    if aggr_on:
        g = glue_stats(report, report.glued_aggr)
        lines.append("[b magenta]— ⊗ при АГРЕССИВНОЙ склейке (через структ. границы, кроме таблиц):[/]")
        if g["n"] == 0:
            lines.append("[dim]   короткой прозы для агр.склейки нет → +0[/]")
        else:
            lines.append(
                f"   [magenta]+{g['susp']} подозрительно из {g['n']} фрагментов[/] "
                f"[dim]({g['frag_chars']} симв короткой прозы)[/]")
            lines.append(
                f"   [b magenta]общая доля ИИ с учётом агр.склейки: {g['with_share']:.1%}[/] "
                f"[dim](без склейки {g['base_share']:.1%})[/]")
        lines.append("[magenta]⊗ Агрессивная склейка игнорирует структурные границы и может "
                     "объединять смысловые блоки. Для закрытия лазеек обхода через нарезку. "
                     "Не используйте как основной режим.[/]")
    for w in report.warnings:
        lines.append(f"[yellow]⚠ {escape(w)}[/]")
    return "\n".join(lines)


class _FilteredTree(DirectoryTree):
    """Дерево показывает только папки и поддерживаемые форматы (.docx/.pdf/.txt)."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [p for p in paths
                if not p.name.startswith(".")
                and (p.is_dir() or p.suffix.lower() in EXTS)]


class FileSelect(Screen):
    """Экран выбора файла: дерево + ручной ввод пути."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[b]Выберите документ для проверки[/]  [dim]— Enter в дереве "
                     "или впишите путь ниже. Форматы: .docx / .pdf / .txt[/]", id="fs-hint")
        yield Input(placeholder="путь к файлу, напр. samples/test_diploma.docx", id="fs-path")
        yield _FilteredTree(str(self.app.start_dir), id="fs-tree")
        yield Footer()

    def on_mount(self) -> None:
        self.app.title = "AI Detector — выбор файла"
        self.app.sub_title = "оболочка поверх бэкенда (ML не меняется)"
        self.query_one("#fs-tree").focus()

    def on_directory_tree_file_selected(self, e: DirectoryTree.FileSelected) -> None:
        self.app.open_file(str(e.path))

    def on_input_submitted(self, e: Input.Submitted) -> None:
        if e.value.strip():
            self.app.open_file(e.value.strip())


class Processing(Screen):
    """«Обрабатываю…» — пока считается перплексия/трансформер (не мгновенно)."""

    def __init__(self, path: str, key: str) -> None:
        super().__init__()
        self._path, self._key = path, key

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="proc-wrap"):
            yield Static(f"⏳  Обрабатываю  [b]{escape(Path(self._path).name)}[/]\n"
                         f"[dim]модель {self._key}: перплексия / трансформер считаются "
                         f"не мгновенно — это не зависание[/]", id="proc-msg")
            yield LoadingIndicator()
        yield Footer()


class ReportScreen(Screen):
    """Главный экран — отчёт: карта абзацев + сводка + дисклеймер."""

    BINDINGS = [
        ("pagedown", "pgdn", "PgDn ▼"),
        ("pageup", "pgup", "PgUp ▲"),
        ("home", "top", "В начало"),
        ("end", "bottom", "В конец"),
        ("g", "toggle_glue", "Склейка прозы"),
        ("G", "toggle_aggressive", "Агр.склейка"),
        ("a", "agreement", "Согласие v3+v4"),
        ("e", "export", "Экспорт отчёта"),
    ]

    def __init__(self, report: Report) -> None:
        super().__init__()
        self.report = report
        self.glue_on = False
        self.aggr_on = False         # агрессивная склейка (через структ. границы) — клавиша G
        self.agree_on = False
        self.agreement = None        # list[AgreeResult] — ленивый кэш (считается по 'a')

    def _map_items(self):
        """Виджеты карты. Приоритет режимов: согласованность (a) → агрессивная склейка (G)
        → обычная склейка (g) → норма. Склейки сворачивают абзацы-члены в один фрагмент
        в позиции первого члена (обычная — _glued_static, агрессивная — _aggr_static)."""
        r = self.report
        if self.agree_on and self.agreement is not None:
            items = [(a.idx, "a", a) for a in self.agreement]
            items += [(pp.idx, "s", (pp, reason)) for pp, reason in r.dropped]
            items.sort(key=lambda x: x[0])
            return [(_agree_static(p) if k == "a" else _service_static(*p)) for _, k, p in items]
        # обычная (g) или агрессивная (G) склейка — общий механизм, разный список+виджет
        frags = r.glued_aggr if self.aggr_on else (r.glued if self.glue_on else [])
        make_glue = _aggr_static if self.aggr_on else _glued_static
        member, anchor = {}, {}
        for f in frags:
            anchor[f.anchor_idx] = f
            for i in f.member_idxs:
                member[i] = f
        items = [(p.idx, "c", p) for p in r.paragraphs]
        items += [(pp.idx, "s", (pp, reason)) for pp, reason in r.dropped]
        items.sort(key=lambda x: x[0])
        widgets = []
        for idx, knd, payload in items:
            if frags and knd == "c" and idx in member:
                if idx in anchor:                 # первый член -> показываем склейку
                    widgets.append(make_glue(anchor[idx]))
                continue                          # прочие члены свёрнуты в склейку
            widgets.append(_para_static(payload) if knd == "c" else _service_static(*payload))
        return widgets

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        r = self.report
        if r.error:
            body = f"[b red]{escape(r.error)}[/]"
            for w in r.warnings:
                body += f"\n[yellow]⚠ {escape(w)}[/]"
            yield Static(body, id="summary")
        else:
            yield Static(_summary_markup(r), id="summary")
            with VerticalScroll(id="map"):
                for w in self._map_items():
                    yield w
        yield Static(DISCLAIMER, id="disclaimer")
        yield Footer()

    def on_mount(self) -> None:
        self.app.title = f"AI Detector — {Path(self.report.path).name}"
        self.app.sub_title = MODEL_SHORT[self.report.model_key]
        try:
            self.query_one("#map").focus()
        except Exception:
            pass
        # смешанный жанр -> сразу включаем режим согласованности (рекомендация классификатора)
        g = self.report.genre
        if g and g.get("recommended") == "agree" and not self.report.error:
            self.action_agreement()

    def _map(self) -> Optional[VerticalScroll]:
        try:
            return self.query_one("#map", VerticalScroll)
        except Exception:
            return None

    def action_pgdn(self) -> None:
        if m := self._map():
            m.scroll_page_down()

    def action_pgup(self) -> None:
        if m := self._map():
            m.scroll_page_up()

    def action_top(self) -> None:
        if m := self._map():
            m.scroll_home()

    def action_bottom(self) -> None:
        if m := self._map():
            m.scroll_end()

    def _rerender(self) -> None:
        """Перерисовать сводку (#summary) и карту (#map) под текущий режим."""
        try:
            if self.agree_on and self.agreement is not None:
                self.query_one("#summary", Static).update(
                    _agreement_summary_markup(self.report, self.agreement))
            else:
                self.query_one("#summary", Static).update(
                    _summary_markup(self.report, self.glue_on, self.aggr_on))
        except Exception:                                           # noqa: BLE001
            pass
        m = self._map()
        if m:
            m.remove_children()
            m.mount(*self._map_items())

    def action_toggle_glue(self) -> None:
        """Диагностика: склеить соседнюю короткую прозу, обновить КАРТУ и СВОДКУ
        (вторая строка «с учётом склейки»). Основную сводку-факт не подменяет."""
        if self.report.error:
            return
        self.glue_on = not self.glue_on
        if self.glue_on:
            self.agree_on = False                # режимы взаимоисключающие
            self.aggr_on = False
        self._rerender()
        if not self.glue_on:
            self.notify("Склейка выкл — карта и сводка как факт сегментации")
            return
        g = glue_stats(self.report)
        if g["n"] == 0:
            self.notify("Склейка ВКЛ: соседней короткой прозы нет → +0 "
                        "(структура документа человекоподобна).")
        else:
            self.notify(f"Склейка ВКЛ: +{g['susp']} подозрит., +{g['uncert']} неопр. "
                        f"из {g['n']} фрагм. → доля подозрит. {g['base_share']:.0%}→{g['with_share']:.0%}. "
                        "Диагностика — основная сводка-факт сохранена.")

    def action_toggle_aggressive(self) -> None:
        """АГРЕССИВНАЯ склейка (G): соседняя короткая проза через структурные границы
        (кроме таблиц). Отдельный диагностический режим, ВЫКЛ по умолчанию — осознанно."""
        if self.report.error:
            return
        self.aggr_on = not self.aggr_on
        if self.aggr_on:
            self.glue_on = False                 # режимы взаимоисключающие
            self.agree_on = False
        self._rerender()
        if not self.aggr_on:
            self.notify("Агр.склейка выкл — обычная карта")
            return
        g = glue_stats(self.report, self.report.glued_aggr)
        if g["n"] == 0:
            self.notify("⊗ Агр.склейка ВКЛ: короткой прозы для склейки нет → +0")
        else:
            self.notify(f"⊗ АГР.склейка ВКЛ (через структ. границы, кроме таблиц): "
                        f"+{g['susp']} подозр. из {g['n']} фрагм. → доля ИИ "
                        f"{g['base_share']:.0%}→{g['with_share']:.0%}. НЕ основной режим — "
                        "может объединять РАЗНЫЕ смысловые блоки.")

    def action_agreement(self) -> None:
        """Согласованность v3+v4: красит абзацы по согласию (ИИ/человек/расхождение).
        Делает выводимым жанровый перекос v4 (расхождение = «перепроверить»)."""
        if self.report.error:
            return
        if self.agree_on:                        # выключить
            self.agree_on = False
            self._rerender()
            self.notify("Согласованность выкл — обычная карта")
            return
        if self.agreement is not None:           # уже посчитано — показать
            self.agree_on = True
            self.glue_on = self.aggr_on = False
            self._rerender()
            return
        self.notify("Считаю согласованность v3+v4… (две модели, ~пара секунд)")
        self.app.compute_agreement(self)         # фон (worker) -> on_agreement_ready

    def on_agreement_ready(self, agreement) -> None:
        self.agreement = agreement
        self.agree_on = True
        self.glue_on = self.aggr_on = False
        self._rerender()
        if not agreement:
            self.notify("Согласованность: нет абзацев с вердиктом")
            return
        c = agreement_counts(agreement)
        self.notify(f"Согласованность: оба-ИИ {c[AGREE_AI]}, оба-человек {c[AGREE_HUMAN]}, "
                    f"РАСХОЖДЕНИЕ {c[AGREE_DISAGREE]} (оранжевые — жанр-чувствительно, перепроверить).")

    def action_export(self) -> None:
        """Сохранить отчёт (markdown + HTML) рядом с документом."""
        if self.report.error:
            self.notify("Нечего экспортировать", severity="warning")
            return
        try:
            paths = export_report_files(self.report, glue_on=self.glue_on)
            extra = " (со склейкой)" if self.glue_on else ""
            self.notify(f"Отчёт сохранён{extra}:\n" + "\n".join(p.name for p in paths)
                        + f"\nв {paths[0].parent}")
        except Exception as e:                                       # noqa: BLE001
            self.notify(f"Ошибка экспорта: {e}", severity="error")


class DetectorApp(App):
    CSS = """
    Screen { background: $surface; }

    #fs-hint { padding: 1 1 0 1; }
    #fs-path { margin: 1 1; }
    #fs-tree { height: 1fr; border: round $primary; margin: 0 1 1 1; }

    #proc-wrap { align: center middle; height: 1fr; }
    #proc-msg { text-align: center; padding: 1 2; }

    #summary { height: auto; padding: 1; margin: 0 1; border: round $primary; }
    #map { height: 1fr; padding: 0 1; }
    #disclaimer { height: auto; padding: 0 1; background: $panel; border-top: solid $primary; }

    .para { height: auto; padding: 0 1; margin: 0 1 1 1; }
    .v-ai          { background: red 12%;    border-left: thick red; }
    .v-uncertain   { background: yellow 12%; border-left: thick yellow; }
    .v-human       { background: green 12%;  border-left: thick green; }
    .v-insufficient{ background: $panel;     border-left: thick $surface-lighten-2; color: $text-muted; }
    .v-service     { background: $panel;     border-left: thick $surface-lighten-1; color: $text-muted; text-style: italic; }
    .glued         { border-left: thick $warning 60%; border-top: dashed $warning 40%; }
    .glued-aggr    { border-left: thick magenta 70%; border-top: dashed magenta 60%; }
    .v-disagree    { background: orange 16%; border-left: thick orange; }
    """

    BINDINGS = [
        ("o", "open_file_screen", "Открыть файл"),
        ("m", "switch_model", "Модель v1→v2→v3→v4"),
        ("q", "quit", "Выход"),
    ]

    def __init__(self, start_dir, auto_open: Optional[str] = None) -> None:
        super().__init__()
        self.start_dir = Path(start_dir)
        self.auto_open = auto_open
        # дефолт карты — v4 (боевая): шире покрытие генераторов (llama/mistral).
        # m переключает v1→v2→v3→v4; v3 остаётся откатом (консервативнее).
        self.model_key = "v4"
        self.current_path: Optional[str] = None
        self._genre: Optional[dict] = None        # вердикт автоклассификатора жанра

    def on_mount(self) -> None:
        self.push_screen(FileSelect())
        if self.auto_open:
            self.open_file(self.auto_open)

    # --- открытие файла / переключение модели ---
    def open_file(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self.notify(f"Файл не найден: {path}", severity="error")
            return
        if p.suffix.lower() not in EXTS:
            self.notify("Поддерживаются только .docx / .pdf / .txt", severity="error")
            return
        self.current_path = str(p)
        self._launch_scan(auto_genre=True)        # жанр+согласие считаем в фоне (не морозим UI)

    def action_switch_model(self) -> None:
        if not self.current_path:
            self.notify("Сначала выберите файл (o)")
            return
        self.model_key = MODELS[(MODELS.index(self.model_key) + 1) % len(MODELS)]
        self.notify(f"Модель → {MODEL_SHORT[self.model_key]} · пересчёт (ручной выбор)")
        self._launch_scan(auto_genre=False)       # ручной выбор: жанр НЕ переопределяет

    def action_open_file_screen(self) -> None:
        self._to_base()                # показать постоянный экран выбора файла

    # --- управление стеком экранов ---
    def _to_base(self) -> None:
        """
        Свести стек к [служебный Screen, FileSelect]. FileSelect держим ПОСТОЯННЫМ
        на дне: у Textual под ним всегда есть собственный базовый Screen (stack[0]),
        поэтому целевая глубина = 2, а не 1 — иначе снесём сам FileSelect.
        """
        while len(self.screen_stack) > 2:
            self.pop_screen()
        if len(self.screen_stack) < 2 or not isinstance(self.screen_stack[1], FileSelect):
            while len(self.screen_stack) > 1:      # страховка: FileSelect потерялся
                self.pop_screen()
            self.push_screen(FileSelect())

    # --- асинхронный прогон, чтобы UI не висел ---
    def _launch_scan(self, auto_genre: bool = False) -> None:
        self.push_screen(Processing(self.current_path, self.model_key))
        self._scan_worker(self.current_path, self.model_key, auto_genre)

    @work(thread=True, exclusive=True)
    def _scan_worker(self, path: str, key: str, auto_genre: bool = False) -> None:
        """В ФОНЕ: при auto_genre — жанр+второй слой согласия (recommend, грузит v3+v4),
        выбор модели; затем скан. UI не морозим (всё в треде). m-переключение — auto_genre
        False: используем заданную модель, жанр НЕ трогаем."""
        genre = self._genre
        if auto_genre:
            genre = None
            try:
                text, _w, _b = extract_blocks(path)
                if text.strip():
                    genre = recommend(text)
            except Exception:                                      # noqa: BLE001
                genre = None
            if genre:
                rec = genre["recommended"]
                key = rec if rec in ("v3", "v4") else "v4"         # mixed/agree -> база v4
        report = run_scan(path, key, genre)
        self.call_from_thread(self._after_scan, report, genre, key, auto_genre)

    def _after_scan(self, report: Report, genre, key: str, auto_genre: bool) -> None:
        self._genre = genre
        self.model_key = key
        if auto_genre and genre:
            rec = genre["recommended"]
            recname = {"v3": "v3", "v4": "v4", "agree": "согласованность (a)"}[rec]
            agr = genre.get("agreement")
            extra = (f" · согласие {'низкое' if agr['div'] > AGREE_DIV_THRESH else 'высокое'} "
                     f"({agr['div']:.0%})") if (agr and agr.get("assessable")) else ""
            if genre.get("overridden"):
                self.notify(f"Жанр: {genre['genre']}, но v3/v4 расходятся{extra} → автосогласованность")
            else:
                self.notify(f"Жанр: {genre['genre']} · рекомендую {recname}{extra}")
        self._present(report)

    def _present(self, report: Report) -> None:
        self._to_base()                            # [Screen, FileSelect]
        self.push_screen(ReportScreen(report))     # -> [Screen, FileSelect, ReportScreen]

    @work(thread=True, exclusive=True, group="agreement")
    def compute_agreement(self, screen) -> None:
        """Фоновый прогон v3+v4 для режима согласованности (две модели — не мгновенно)."""
        try:
            agreement, _dropped, _warn = agreement_results(screen.report.path)
        except Exception as e:                                      # noqa: BLE001
            self.call_from_thread(self.notify, f"Ошибка согласованности: {e}", severity="error")
            return
        self.call_from_thread(screen.on_agreement_ready, agreement)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Терминальный TUI проверки документов на ИИ.")
    ap.add_argument("start", nargs="?", default=None,
                    help="папка для файлового дерева или файл (по умолчанию samples/)")
    ap.add_argument("--text", metavar="FILE",
                    help="headless: напечатать текстовый отчёт и выйти (без TUI)")
    ap.add_argument("--model", choices=["v1", "v2", "v3", "v4"], default="v4",
                    help="модель для --text (по умолчанию v4 — боевая, как и карта в TUI)")
    ap.add_argument("--export", metavar="FILE",
                    help="headless: проверить FILE и сохранить отчёт (.md + .html) рядом")
    ap.add_argument("--glue", action="store_true",
                    help="включить диагностику склейки короткой прозы (для --text/--export)")
    ap.add_argument("--agree", metavar="FILE",
                    help="headless: согласованность v3+v4 по FILE (где модели расходятся)")
    ap.add_argument("--genre", metavar="FILE",
                    help="headless: определить жанр FILE и рекомендуемую модель")
    args = ap.parse_args()

    if args.genre:
        text, _w = extract_text(Path(args.genre))
        r = recommend(text) if text.strip() else None       # жанр + второй слой согласия
        if not r:
            print("[!] текста для классификации нет")
            return
        print(f"Жанр: {r['genre']} · рекомендую {r['recommended']}")
        print(f"  {r['reason']}")
        agr = r.get("agreement")
        if agr and agr.get("assessable"):
            lvl = "низкое" if agr["div"] > AGREE_DIV_THRESH else "высокое"
            print(f"  согласие моделей: {lvl} ({agr['div']:.0%} сильно-расходящихся окон из {agr['n']})")
        print(f"  tech_signal={r['tech_signal']} hum_signal={r['hum_signal']} confident={r['confident']} "
              f"overridden={r.get('overridden')}")
        return

    if args.agree:
        print(render_agreement_ansi(Path(args.agree)))
        return

    if args.export:
        report = run_scan(Path(args.export), args.model)
        if report.error:
            print(f"[!] {report.error}", file=sys.stderr)
            return
        for p in export_report_files(report, glue_on=args.glue):
            print(f"[сохранено] {p}")
        return

    if args.text:
        report = run_scan(Path(args.text), args.model)
        print(render_ansi(report, args.model, glue_on=args.glue))
        return

    start = Path(args.start) if args.start else (Path("samples") if Path("samples").is_dir()
                                                 else Path("."))
    auto_open = None
    if start.is_file():
        auto_open = str(start)
        start = start.parent
    DetectorApp(start, auto_open=auto_open).run()


if __name__ == "__main__":
    main()
