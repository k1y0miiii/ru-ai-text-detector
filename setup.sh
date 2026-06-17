#!/usr/bin/env bash
# Установка окружения и моделей.
#   ./setup.sh           — venv + зависимости + боевая модель v4 (для запуска)
#   ./setup.sh --train   — то же + базовые модели для переобучения (с HuggingFace)
set -euo pipefail
cd "$(dirname "$0")"

# HF Model-репо с весами ВСЕХ моделей (v1–v4), структурой models/. Поменяйте, если
# форкнули. revision пиннится на конкретный коммит (воспроизводимость + защита от
# подмены — калибраторы грузятся joblib.load).
MODELS_REPO="k1y0mi/ru-ai-text-detector"
MODELS_REV="175e75ec79e9471f3f5b4ab0da9b859188446807"

# Зависимости (torch 2.4.1 / sklearn 1.5.2 / numpy 1.26.4) выпущены до Python 3.13
# и не имеют под него колёс; модели models/*.joblib пиклены sklearn 1.5.2, так что
# версию стека не трогаем. Значит, нужен Python 3.10–3.12. По умолчанию берём
# python3.12 (если есть), иначе python3 — но всё равно проверяем версию ниже.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python3.12 >/dev/null 2>&1; then PY=python3.12; else PY=python3; fi
fi

# Гард: pinned-стек не ставится на 3.13+. Проверяем интерпретатор, который реально
# будет использован (готовый venv — его python; иначе — выбранный $PY).
require_py() {
  local v; v="$("$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")"
  case "$v" in
    3.10|3.11|3.12) return 0 ;;
    *) echo "ОШИБКА: нужен Python 3.10–3.12 (у '$1' версия $v)." >&2
       echo "       Стек (torch/sklearn/numpy) не имеет колёс под Python 3.13+." >&2
       echo "       Установите 3.12 и запустите: PYTHON=python3.12 ./setup.sh" >&2
       return 1 ;;
  esac
}

echo "[1/3] venv"
if [ ! -d venv ]; then
  require_py "$PY"
  "$PY" -m venv venv
else
  require_py venv/bin/python
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip >/dev/null

echo "[2/3] зависимости и пакет (editable)"
pip install -e .

echo "[3/3] модели v1–v4 из HF Model-репо -> models/"
# Кладём ВСЕ модели в models/ той структурой, что ждут transformer_score.py / app.py /
# CLI / TUI: models/v2_model models/v3_model models/v4_model models/model.joblib
# models/threshold_*.json models/*_calibrator.joblib.
if [ -f models/v4_model/model.safetensors ]; then
  echo "      уже на месте — пропускаю"
else
  python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODELS_REPO}', repo_type='model', revision='${MODELS_REV}', local_dir='models')"
  rm -rf models/.cache
fi
# Модель перплексии для v1 (ai-forever/rugpt3small) transformers скачает сама
# при первом запуске — отдельный шаг не нужен.

if [ "${1:-}" = "--train" ]; then
  echo "[train] базовая модель для дообучения (ai-forever/ruBert-base)"
  pip install -U "huggingface_hub[cli]" >/dev/null
  mkdir -p bases
  huggingface-cli download ai-forever/ruBert-base --local-dir bases/ruBert-base
  # e5-small нужен только для probe_base.py (эксперимент), ставится по желанию:
  # huggingface-cli download intfloat/multilingual-e5-small --local-dir bases/e5-small
fi

echo
echo "Готово. Запуск:"
echo "  source venv/bin/activate"
echo "  python apps/detect_cli.py samples/sample_ai_1.txt"
