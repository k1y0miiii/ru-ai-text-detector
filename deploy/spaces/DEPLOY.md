# Деплой на Hugging Face Spaces (Docker SDK)

Пошагово: как поднять веб-детектор на HF Spaces так, чтобы он **не отваливался**,
давал **выбор всех моделей** и был **под телефон**.

## Что уже готово в репозитории

- `Dockerfile` — Docker SDK, порт **8000**; прогревает кэш модели перплексии v1
  (`rugpt3small`) на этапе сборки и включает офлайн-режим (`HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`) — в рантайме сеть к huggingface.co не нужна.
- `apps/app.py` — `GET /` отдаёт мобильную страницу с выбором модели; `POST /detect`
  как прежде.
- `deploy/spaces/README.md` — README **Space-репо** с метаданными (`sdk: docker`,
  `app_port: 8000`). На Spaces его нужно положить в корень репозитория Space.

## Предусловия

- Аккаунт на huggingface.co и **Access Token** с правом записи:
  <https://huggingface.co/settings/tokens>.
- Установлены `git` и `git-lfs`:
  ```bash
  git lfs install
  ```
- Веса моделей лежат локально в `models/` (v2_model, v3_model, v4_model,
  калибраторы, `threshold_*.json`, `model.joblib`). Суммарно ~1.6 ГБ.

## Шаг 1. Создать Space

На huggingface.co → **New Space** → SDK = **Docker** (пустой/Blank). Получите URL
вида `https://huggingface.co/spaces/<user>/<space>`.

## Шаг 2. Подключить Space как git-remote

Из корня этого репозитория:

```bash
git remote add space https://huggingface.co/spaces/<user>/<space>
# логин/пароль при push: username = ваш ник, password = Access Token
```

## Шаг 3. Положить метаданные Space в корень

HF читает YAML-заголовок из корневого `README.md` Space-репо. Скопируйте туда наш
файл (он перетрёт README проекта ТОЛЬКО в ветке, которую пушим в Space, — удобно
держать отдельную ветку для деплоя):

```bash
git checkout -b space-deploy
cp deploy/spaces/README.md README.md
```

## Шаг 4. Затрекать веса через git-lfs

Модели большие — только через LFS, иначе push отклонят:

```bash
git lfs track "models/**"
git add .gitattributes
git add models README.md
git commit -m "deploy: HF Space (Docker) с весами через git-lfs"
```

## Шаг 5. Push в Space

```bash
git push space space-deploy:main
```

HF соберёт Docker-образ (прогрев `rugpt3small` идёт на этом этапе — нужна сеть на
билд-машине HF, она есть) и поднимет сервис на порту 8000. Первый билд из-за весов
и зависимостей torch занимает несколько минут.

## Шаг 6. Проверка

- Открыть `https://<user>-<space>.hf.space/` — страница детектора.
- `https://<user>-<space>.hf.space/health` → `{"status":"ok", ...}`.
- Прогнать текст на каждой модели (v1–v4).

## Заметки

- **«Отвалился» = краш загрузки модели** — закрыто: веса v2–v4 в образе (через LFS),
  `rugpt3small` запечён в кэш, рантайм офлайн. HF-аптайм не нужен для инференса.
- **«Отвалился» = Space уснул** — бесплатный Space засыпает после ~48 ч простоя.
  Это поведение тарифа, кодом не лечится. Снизить простой — keep-alive
  (`.github/workflows/keepalive.yml` пингует `/health`; задайте repo-variable
  `SPACE_URL`). Гарантированный аптайм — только платный тариф Spaces.
- Обновление: повторить шаги 3–5 (или просто `git push space …` после новых
  коммитов в `space-deploy`).
