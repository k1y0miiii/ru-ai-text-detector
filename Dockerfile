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

# Веса трансформера v4 (models/v4_model/) в образ не кладутся — смонтируйте
# каталог models/ томом или прогоните setup.sh до сборки. Модель перплексии для
# v1 (ai-forever/rugpt3small) скачается в кэш при первом запросе; для прод-образа
# кэш лучше прогреть на этапе сборки (см. README).

EXPOSE 8000
CMD ["uvicorn", "apps.app:app", "--host", "0.0.0.0", "--port", "8000"]
