#!/usr/bin/env python3
"""
verify_window.py — проверка смены дефолта окна 300->150 на реальных docx.

Показывает 300 vs 150 бок о бок на обеих моделях и обоих порогах (вердиктный
0.86/0.90 и мягкий порог ПОДСВЕТКИ карты 0.60/0.55). Цель: (1) ИИ-вставки в
mixed/pure_ai не ХУЖЕ прежнего; (2) человек (diploma + человеческие абзацы mixed)
чист — FPR не прыгает. Ничего не дообучаем, только гоняем скан с двумя окнами.
"""

from pathlib import Path

import measure_window as M   # переиспуем scan() и опознание ИИ-вставок test_mixed
from aidetector.extract_text import extract_text

VERDICT = {"v1": 0.86, "v2": 0.90}
HIGHLIGHT = {"v1": 0.60, "v2": 0.55}

# документы и их «природа» содержательных абзацев
DOCS = [
    ("test_mixed.docx", "mixed"),       # вставки ИИ среди человека (ключ test_key.md)
    ("test_pure_ai.docx", "all_ai"),    # весь контент — ИИ
    ("test_ai_edited.docx", "all_ai"),  # ИИ с человеческой редактурой (слепое пятно)
    ("test_diploma.docx", "all_human"), # 100% человек
]


def metrics(path, kind, model, W, thr):
    """-> (ai_flag, ai_total, hum_flag, hum_total) по содержательным абзацам с вердиктом."""
    text, _ = extract_text(Path("samples") / path)
    res = M.scan(text, model, W, thr)
    af = at = hf = ht = 0
    for r in res:
        flagged = (r.label == "подозрительно")
        if kind == "mixed":
            k = M._ai_kind(r.text)
            if k in ("long", "short"):
                at += 1; af += int(flagged)
            elif r.p_ai is not None:
                ht += 1; hf += int(flagged)
        elif kind == "all_ai":
            if r.p_ai is not None:
                at += 1; af += int(flagged)
        elif kind == "all_human":
            if r.p_ai is not None:
                ht += 1; hf += int(flagged)
    return af, at, hf, ht


def run(thr_map, title):
    print("\n" + "=" * 78)
    print(f"  {title}   v1={thr_map['v1']}  v2={thr_map['v2']}")
    print("=" * 78)
    print(f"  {'документ':18s} {'модель':>6} | {'recall ИИ  300сл->150сл':>26} | "
          f"{'FPR люди  300сл->150сл':>24}")
    for path, kind in DOCS:
        for model in ("v1", "v2"):
            a3, at3, h3, ht3 = metrics(path, kind, model, 300, thr_map[model])
            a1, at1, h1, ht1 = metrics(path, kind, model, 150, thr_map[model])
            ai = (f"{a3}/{at3} -> {a1}/{at1}" if at3 else "—") if kind != "all_human" else "—"
            fp = (f"{h3}/{ht3} -> {h1}/{ht1}" if ht3 else "—") if kind != "all_ai" else "—"
            print(f"  {path:18s} {model:>6} | {ai:>26} | {fp:>24}")


def main():
    run(VERDICT, "ВЕРДИКТНЫЙ порог (обвинение)")
    run(HIGHLIGHT, "Порог ПОДСВЕТКИ карты в TUI")
    print("\nЧтение: recall ИИ слева->справа = было(300)->стало(150) (больше=лучше, "
          "'не хуже'=не упало). FPR люди: 0 или почти 0 = человек чист; рост = цена окна.")


if __name__ == "__main__":
    main()
