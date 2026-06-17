# Деплой на Hugging Face Spaces (Docker SDK)

Как поднять веб-детектор на HF Spaces так, чтобы он **не отваливался**, давал
**выбор всех моделей** и был **под телефон**.

## Архитектура деплоя

У бесплатных **Spaces лимит репозитория 1 ГБ** — все веса (~1.5 ГБ) туда не влезают.
Поэтому:

- **Веса** (v1–v4) лежат в отдельном **Model-репо** `k1y0mi/ru-ai-text-detector`
  (у model-репо лимита 1 ГБ нет). Структура зеркалит `models/`:
  `v2_model/ v3_model/ v4_model/` + `*_calibrator.joblib` + `threshold_*.json` +
  `model.joblib` (v1).
- **Space** `k1y0mi/ru-ai-text-detector` (Docker SDK) — только код + `Dockerfile`,
  репозиторий маленький. На этапе **build** Dockerfile скачивает веса из Model-репо
  (`snapshot_download`) и запекает в образ. В **рантайме** включён офлайн
  (`HF_HUB_OFFLINE=1`) — сеть к huggingface.co не нужна, сервис не падает.

То есть build-time зависимость от HF есть (один раз, при сборке), а рантайм — нет.

## Что уже готово в репозитории

- `Dockerfile` — Docker SDK, порт 8000; качает веса из Model-репо на build
  (если `models/` уже в контексте — например, self-hosted через `setup.sh` — шаг
  пропускается), прогревает кэш `rugpt3small`, включает офлайн-рантайм.
- `apps/app.py` — `GET /` отдаёт мобильную страницу с выбором модели; `POST /detect`
  без изменений.
- `deploy/spaces/README.md` — README **Space-репо** с метаданными (`sdk: docker`,
  `app_port: 8000`).

## Обновление весов в Model-репо (когда модели поменялись)

Залить локальный `models/` (нужную структуру) в Model-репо:

```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="models",                 # или подготовленный staging
    repo_id="k1y0mi/ru-ai-text-detector",
    repo_type="model",
    delete_patterns="*",                  # чистый mirror
)
```

(Нужен `huggingface-cli login` с токеном на запись. Веса крупные — заливаются через
LFS автоматически; одинаковые файлы HF дедуплицирует по хэшу.)

## Деплой/обновление Space (код)

Залить код Space (без весов — они придут на build):

```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path=".",                      # или staging только с кодом+Dockerfile
    repo_id="k1y0mi/ru-ai-text-detector",
    repo_type="space",
    delete_patterns="*",
    ignore_patterns=["models/**", "venv/**", ".git/**", "data/**",
                     "training/**", "evaluation/**", "samples/**",
                     "CLAUDE.md", ".claude/**", "docs/**", "tests/**"],
    commit_message="update space",
)
```

README с `sdk: docker` переключает Space на Docker автоматически. После пуша HF
пересобирает образ (несколько минут: torch + скачивание весов + прогрев).

## Проверка

- `https://k1y0mi-ru-ai-text-detector.hf.space/` — страница детектора.
- `.../health` → `{"status":"ok", ...}`.
- Прогнать текст на каждой модели (v1–v4).

## Заметки

- **«Отвалился» = краш загрузки модели** — закрыто: веса в образе (скачаны на build),
  `rugpt3small` запечён, рантайм офлайн.
- **«Отвалился» = Space уснул** — бесплатный Space засыпает после ~48 ч простоя
  (поведение тарифа, кодом не лечится). Снизить простой — keep-alive
  (`.github/workflows/keepalive.yml` пингует `/health`; задайте repo-variable
  `SPACE_URL`). Гарантированный аптайм — только платный тариф.
