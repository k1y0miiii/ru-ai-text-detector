FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Зависимости отдельным слоем (кэшируется, пока не менялись requirements/pyproject)
COPY requirements.txt pyproject.toml ./
COPY aidetector ./aidetector
RUN pip install --no-cache-dir -e .

COPY . .

# Веса всех моделей (v1–v4). Если models/ уже в контексте (self-hosted: положили через
# setup.sh) — шаг пропускается. Иначе (HF Spaces: у Space-репо лимит 1 ГБ, веса туда не
# кладём) скачиваем из публичного Model-репо НА ЭТАПЕ BUILD и запекаем в образ. Структура
# Model-репо зеркалит models/ (v2_model/ v3_model/ v4_model/ + калибраторы + пороги).
# revision пиннится на конкретный коммит Model-репо: воспроизводимая сборка + защита
# от молчаливой подмены весов на плавающем main (артефакты first-party, но joblib.load
# калибраторов — десериализация, поэтому фиксируем источник по SHA).
ARG MODELS_REV=175e75ec79e9471f3f5b4ab0da9b859188446807
RUN python -c "import os; from huggingface_hub import snapshot_download; (print('models present, skip') if os.path.exists('models/v4_model/model.safetensors') else snapshot_download('k1y0mi/ru-ai-text-detector', repo_type='model', revision='${MODELS_REV}', local_dir='models'))" \
    && rm -rf models/.cache

# Прогрев кэша модели перплексии v1 (ai-forever/rugpt3small) ПРЯМО В ОБРАЗ. Нужна на
# КАЖДЫЙ запрос (app.py всегда считает фичи v1), поэтому без неё /detect не работает.
# Качается здесь — на этапе build есть сеть; в рантайме её уже не требуется.
RUN python -c "import aidetector.features as f; f._lazy_load()"

# Рантайм — полностью офлайн: from_pretrained берёт веса с диска и НЕ ходит на
# huggingface.co. Так сервис не «отваливается», когда HF недоступен. ВАЖНО: эти ENV
# идут ПОСЛЕ прогрева выше (иначе скачивание упало бы).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
CMD ["uvicorn", "apps.app:app", "--host", "0.0.0.0", "--port", "8000"]
