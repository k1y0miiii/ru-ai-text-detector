"""Пути к артефактам модели и данным.

Точки входа и обучающие скрипты находят models/ и data/ независимо от того, из
какого каталога их запустили. Корень проекта вычисляется от расположения пакета;
переопределяется переменной окружения AIDETECTOR_ROOT (если держите веса в другом
месте).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("AIDETECTOR_ROOT") or Path(__file__).resolve().parent.parent)
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = ROOT / "samples"


def model_path(*parts: str) -> str:
    """Абсолютный путь к артефакту в models/ (str — для from_pretrained/joblib)."""
    return str(MODELS_DIR.joinpath(*parts))


def data_path(*parts: str) -> str:
    """Абсолютный путь к файлу в data/."""
    return str(DATA_DIR.joinpath(*parts))
