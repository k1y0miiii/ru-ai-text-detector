#!/usr/bin/env bash
# Установка окружения и моделей.
#   ./setup.sh           — venv + зависимости + боевая модель v4 (для запуска)
#   ./setup.sh --train   — то же + базовые модели для переобучения (с HuggingFace)
set -euo pipefail
cd "$(dirname "$0")"

# Репозиторий и релиз, откуда берём веса v4. Поменяйте, если форкнули.
REPO="k1y0miiii/ru-ai-text-detector"
RELEASE_TAG="v1.0"
V4_ASSET="v4_model.tar.gz"

PY="${PYTHON:-python3}"

echo "[1/3] venv"
if [ ! -d venv ]; then "$PY" -m venv venv; fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip >/dev/null

echo "[2/3] зависимости и пакет (editable)"
pip install -e .

echo "[3/3] боевая модель v4"
if [ -f models/v4_model/model.safetensors ]; then
  echo "      уже на месте — пропускаю"
else
  mkdir -p models
  URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${V4_ASSET}"
  echo "      качаю ${URL}"
  curl -L --fail -o "/tmp/${V4_ASSET}" "${URL}"
  tar -xzf "/tmp/${V4_ASSET}" -C models/
  rm -f "/tmp/${V4_ASSET}"
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
