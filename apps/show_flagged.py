#!/usr/bin/env python3
"""show_flagged.py <docx> — СОДЕРЖИМОЕ помеченных абзацев (v3), не только статистика.
Метки карты по мягкому порогу подсветки (v3=0.6); вердиктный 0.846 отмечаем отдельно.
Печатает: подозрительно (p≥0.6), неопределённо (0.5–0.6), разбивку «недостаточно текста»
по категориям с примерами, и хвост «человек» (топ p<0.5). Модель/сегментацию не трогаем."""

import re
import sys

import numpy as np

from aidetector import document_scan as ds
from aidetector import transformer_score as TF
from aidetector.extract_text import extract_text

HL = 0.6          # порог ПОДСВЕТКИ карты v3 (как в TUI)
VERDICT = 0.846   # боевой порог v3 (длинные >=40 слов)


def flat(t):
    return re.sub(r"\s+", " ", t).strip()


def cat_short(t):
    """Категория короткого абзаца (для «недостаточно текста»)."""
    s = t.strip().lower()
    if re.match(r"^(глава|раздел|выводы|заключение|введение)\b", s) or re.match(r"^\d+(\.\d+)*\.?\s+\S", s):
        return "заголовок/подзаголовок"
    if re.match(r"^(рис\.?|рисунок|табл\.?|таблица|схема|диаграмма|листинг|формула)\b", s):
        return "подпись к рис./табл."
    if re.match(r"^[–—•·*\-]\s", t.strip()) or re.match(r"^[a-zа-я]\)\s", s) or re.match(r"^\d+[.)]\s", t.strip()):
        return "пункт списка"
    if ("=" in t or "∑" in t or "×" in t) and len(t) < 130:
        return "формула/обозначение"
    return "короткая проза"


def scan(path):
    text, _ = extract_text(path)
    paras = ds.split_paragraphs(text)
    split = ds.classify_sections(paras)
    stream, spans = ds._build_content_stream(split.content)
    wins = ds.build_windows(stream)
    if wins:
        for w, p in zip(wins, TF.get("v3").proba([stream[w.cstart:w.cend] for w in wins])):
            w.p_ai = float(p)
    res = ds.project_to_paragraphs(split.content, spans, wins, HL)
    return paras, split, res


def report(path):
    print("\n" + "#" * 88)
    print(f"#  {path}")
    print("#" * 88)
    paras, split, res = scan(path)
    verdicted = [r for r in res if r.p_ai is not None]
    insufficient = [r for r in res if r.p_ai is None]
    tables = [(p, rr) for p, rr in split.dropped if "таблица" in rr]
    print(f"абзацев всего {len(paras)} | проза-content {len(split.content)} | "
          f"с вердиктом {len(verdicted)} | «недостаточно текста» {len(insufficient)} | "
          f"таблиц отсеяно {len(tables)}")

    susp = sorted([r for r in verdicted if r.p_ai >= HL], key=lambda r: -r.p_ai)
    uncert = sorted([r for r in verdicted if 0.5 <= r.p_ai < HL], key=lambda r: -r.p_ai)

    print(f"\n{'='*88}\n1) ПОДОЗРИТЕЛЬНО — p≥{HL} (подсветка карты); ★ = ещё и ≥{VERDICT} (вердиктный): {len(susp)} абз.\n{'='*88}")
    for r in susp:
        star = " ★ВЕРДИКТ" if r.p_ai >= VERDICT else ""
        print(f"\n— абз.{r.idx}  p(ИИ)={r.p_ai:.3f}  ({r.n_chars} симв){star}\n  {flat(r.text)}")

    print(f"\n{'='*88}\n2) НЕОПРЕДЕЛЁННО — 0.5≤p<{HL}: {len(uncert)} абз.\n{'='*88}")
    for r in uncert:
        print(f"\n— абз.{r.idx}  p(ИИ)={r.p_ai:.3f}  ({r.n_chars} симв)\n  {flat(r.text)}")

    print(f"\n{'='*88}\n3) «НЕДОСТАТОЧНО ТЕКСТА» ({len(insufficient)} абз.) — разбивка по категориям\n{'='*88}")
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in insufficient:
        buckets[cat_short(r.text)].append(r)
    for cat in sorted(buckets, key=lambda c: -len(buckets[c])):
        items = buckets[cat]
        print(f"\n  [{cat}] — {len(items)} абз. (примеры):")
        for r in items[:8]:
            print(f"    абз.{r.idx:>4} ({r.n_chars:>3}симв): {flat(r.text)[:96]}")
    print(f"\n  [для сверки] таблицы исключены ОТДЕЛЬНО ({len(tables)}), примеры маркеров:")
    for p, _ in tables[:5]:
        print(f"    абз.{p.idx:>4}: {flat(p.text)[:70]}")

    print(f"\n{'='*88}\n4) ХВОСТ «ЧЕЛОВЕК» — топ-5 по p (но <0.5, метка «человек»)\n{'='*88}")
    for r in sorted([r for r in verdicted if r.p_ai < 0.5], key=lambda r: -r.p_ai)[:5]:
        print(f"\n— абз.{r.idx}  p(ИИ)={r.p_ai:.3f}  ({r.n_chars} симв)\n  {flat(r.text)[:200]}")

    # для сравнения версий — карта (текст→p)
    return {flat(r.text): r.p_ai for r in verdicted}


def compare(path_a, path_b):
    from pathlib import Path
    print("Сравнение пометок: кривой оригинал vs причёсанная ГОСТ-версия")
    ma = report(Path(path_a))
    mb = report(Path(path_b))
    print("\n" + "#" * 88 + "\n#  СВОДКА СРАВНЕНИЯ (по совпадающему ТЕКСТУ абзаца)\n" + "#" * 88)
    common = set(ma) & set(mb)
    only_a = set(ma) - set(mb)
    only_b = set(mb) - set(ma)
    print(f"абзацев с вердиктом: оригинал {len(ma)}, ГОСТ {len(mb)}, текст совпал {len(common)}")
    if common:
        d = np.array([abs(ma[t] - mb[t]) for t in common])
        print(f"|Δp| по совпавшим: медиана {np.median(d):.3f}  макс {d.max():.3f}  "
              f"(идентичный текст → p должно совпасть; сдвиг = разная нарезка окон)")
        big = sorted(((abs(ma[t]-mb[t]), t) for t in common), reverse=True)[:5]
        print("  наибольшие расхождения p:")
        for dd, t in big:
            print(f"    Δ={dd:.3f}  orig={ma[t]:.3f} гост={mb[t]:.3f}  «{t[:60]}»")
    # подозрительные (p>=0.6), которые есть в одной версии и нет в другой
    sa = {t for t in ma if ma[t] >= HL}; sb = {t for t in mb if mb[t] >= HL}
    print(f"\nподозрительных (p≥{HL}): оригинал {len(sa)}, ГОСТ {len(sb)}, совпало {len(sa&sb)}")
    for t in sa - sb:
        print(f"  только в ОРИГИНАЛЕ: p={ma[t]:.3f} «{t[:60]}» (в ГОСТ: "
              f"{mb.get(t, '—текста нет—') if not isinstance(mb.get(t),float) else round(mb[t],3)})")
    for t in sb - sa:
        print(f"  только в ГОСТ: p={mb[t]:.3f} «{t[:60]}»")


if __name__ == "__main__":
    from pathlib import Path
    if len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])
    else:
        report(Path(sys.argv[1]))
