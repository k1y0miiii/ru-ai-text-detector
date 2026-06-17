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

# Веса трансформеров v2–v4 (models/) для self-hosted образа смонтируйте томом или
# прогоните setup.sh до сборки. Для HF Spaces — коммитятся в Space-репо через git-lfs
# (см. deploy/spaces/DEPLOY.md), тогда COPY . . затягивает их в образ.

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
