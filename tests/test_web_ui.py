"""Смоук веб-UI: проверяем проводку маршрутов и содержимое страницы.

Не поднимаем lifespan и не грузим трансформеры (713 МБ) — только импорт app
(маршруты регистрируются на импорте) и чтение шаблона как файла.
"""

from pathlib import Path

from apps.app import app

ROOT = Path(__file__).resolve().parent.parent


def test_routes_registered():
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/" in paths            # веб-страница
    assert "/static" in paths      # mount статики
    assert "/detect" in paths      # старый эндпоинт цел
    assert "/health" in paths      # keep-alive дёргает его


def test_index_has_disclaimer_and_models():
    html = (ROOT / "apps" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "не доказательство авторства" in html      # обязательный дисклеймер
    for v in ("v1", "v2", "v3", "v4"):
        assert f'value="{v}"' in html                  # все 4 модели выбираемы
