"""
Оценка модели на КОРОТКИХ текстах (<200 символов) — это известное слабое место
feature-based детектора (burstiness/перплексия шумят на одном-двух предложениях).

Честность пробника: обучение шло на текстах >=200 символов, поэтому ЛЮБОЙ
текст <200 из coat гарантированно НЕ участвовал в обучении — утечки нет.

Меряем:
  - recall по ИИ на коротких (доля коротких машинных, пойманных как ИИ)
  - FPR на коротких людях (доля коротких человеческих, ошибочно как ИИ)
и сравниваем recall на коротких с recall на длинных (тест), чтобы понять,
осталось ли короткое слабым местом.

Запуск: python eval_short.py [model.joblib] [N_на_класс]
"""

import sys

import joblib
import numpy as np

from aidetector.features import extract, to_vector
from aidetector.paths import model_path

MODEL = sys.argv[1] if len(sys.argv) > 1 else model_path("model.joblib")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
random_seed = 42


def main():
    from datasets import load_dataset
    clf = joblib.load(MODEL)

    ds = load_dataset("RussianNLP/coat", "binary", split="train")  # без streaming
    rng = np.random.default_rng(random_seed)

    # собираем короткие (50..199 символов) тексты обоих классов
    short = {0: [], 1: []}
    order = rng.permutation(len(ds))
    for i in order:
        r = ds[int(i)]
        t = (r["text"] or "").strip()
        if 50 <= len(t) < 200:
            lab = int(r["label"])
            if len(short[lab]) < N:
                short[lab].append(t)
        if len(short[0]) >= N and len(short[1]) >= N:
            break

    print(f"короткие тексты (50..199 симв): человек={len(short[0])}, ИИ={len(short[1])}")

    def predict(texts):
        proba = []
        for t in texts:
            v = to_vector(extract(t))
            proba.append(float(clf.predict_proba([v])[0][1]))
        return np.array(proba)

    p_ai = predict(short[1])
    p_hu = predict(short[0])

    recall_ai = float((p_ai >= 0.5).mean())     # машинные, пойманные как ИИ
    fpr_human = float((p_hu >= 0.5).mean())      # люди, ошибочно как ИИ

    print("\n=== КОРОТКИЕ ТЕКСТЫ (<200 симв) ===")
    print(f"recall по ИИ : {recall_ai:.3f}  (поймано {int((p_ai>=0.5).sum())} из {len(p_ai)})")
    print(f"FPR (люди)   : {fpr_human:.3f}  (ложно {int((p_hu>=0.5).sum())} из {len(p_hu)})")
    print(f"средняя p(ИИ): машина={p_ai.mean():.3f}, человек={p_hu.mean():.3f}")


if __name__ == "__main__":
    main()
