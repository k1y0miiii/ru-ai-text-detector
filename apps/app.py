"""
FastAPI-сервис детектора ИИ-текста.

POST /detect  {"text": "..."}   [?model=v1|v2|v3|v4]
  -> p(ИИ) выбранной модели + вердикт + разбивка по фичам v1 (прозрачность).

Боевая модель — v4 (ruBert-base, переобученный с ИИ Llama3.1/Mistral, см.
transformer_score.py): шире покрытие генераторов. v3 (прежняя боевая, консервативнее)
и v2 (tiny2) доступны как откат через ?model=v3|v2; v1 (признаки) — быстрый и
интерпретируемый.

Отдаём не голый вердикт, а сигналы: фичи v1 показываем ВСЕГДА (даже при вердикте v3) —
это и честнее, и объясняет юзеру, ПОЧЕМУ текст так оценён. Порог решения калиброван
под FPR<=3% (редко ложно обвиняем); на коротких фрагментах порог строже (length-aware).
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from aidetector import transformer_score as TF
from aidetector.features import FEATURE_NAMES, extract, to_vector
from aidetector.paths import model_path

V1_MODEL_PATH = model_path("model.joblib")
V1_THRESHOLD = 0.86          # v1 калиброван под FPR<=3% (threshold_v1.json)
MIN_RELIABLE_CHARS = 200     # ниже — любой детектор на коротком менее надёжен
DEFAULT_MODEL = "v4"
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["v1"] = joblib.load(V1_MODEL_PATH)
    except FileNotFoundError:
        _state["v1"] = None
    # прогреваем боевой трансформер (дефолт v4), чтобы ПЕРВЫЙ запрос не лагал на загрузке
    try:
        TF.get(DEFAULT_MODEL)._ensure()
        _state["default_ok"] = True
    except Exception as e:
        _state["default_ok"], _state["default_err"] = False, str(e)
    yield
    _state.clear()


app = FastAPI(title="AI Text Detector (боевая модель v4)", lifespan=lifespan)

# --- веб-страница (тонкий клиент к /detect; ML-логику ниже НЕ трогаем) ---
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Мобильная веб-страница: вставил текст -> выбрал модель -> вердикт."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "default_model": DEFAULT_MODEL})


class DetectIn(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для проверки")


class DetectOut(BaseModel):
    model: str
    ai_probability: float
    verdict: str
    threshold: float
    features: dict
    note: str


@app.get("/health")
def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL,
            "v1_loaded": _state.get("v1") is not None,
            "default_ready": _state.get("default_ok", False)}


@app.post("/detect", response_model=DetectOut)
def detect(inp: DetectIn, model: str = Query(DEFAULT_MODEL, pattern="^(v1|v2|v3|v4|v3_legacy)$")):
    text = inp.text
    vec = to_vector(extract(text))                       # фичи v1 — для прозрачности ВСЕГДА
    feat_dict = dict(zip(FEATURE_NAMES, [round(float(v), 3) for v in vec]))
    nwords = TF.word_count(text)

    if model == "v1":
        clf = _state.get("v1")
        if clf is None:
            return DetectOut(model="v1", ai_probability=-1.0, verdict="модель не обучена",
                             threshold=V1_THRESHOLD, features=feat_dict,
                             note="Нет model.joblib — обучи v1 (finalize_v1.py).")
        proba = float(clf.predict_proba([vec])[0][1])
        thr, base = V1_THRESHOLD, V1_THRESHOLD
    else:
        sc = TF.get(model)
        proba = float(sc.proba([text])[0])
        thr, base = sc.threshold_for(nwords), sc.threshold()

    verdict = "вероятно ИИ" if proba >= thr else "вероятно человек"

    note = ("Вероятностная оценка, не доказательство авторства. Недопустимо как основание "
            f"для дисциплинарных решений. Модель {model}, порог ИИ={thr:.2f} калиброван "
            "под FPR<=3% (редкие ложные обвинения, часть ИИ пропускается).")
    if thr > base:
        note = (f"[короткий фрагмент: {nwords} сл — применён строже порог {thr:.2f}, "
                "чтобы удержать FPR<=3%] ") + note
    if len(text) < MIN_RELIABLE_CHARS:
        note = (f"[!] Текст короче {MIN_RELIABLE_CHARS} символов — на коротком любой "
                "детектор менее надёжен. " + note)

    return DetectOut(model=model, ai_probability=round(proba, 3), verdict=verdict,
                     threshold=round(thr, 3), features=feat_dict, note=note)
