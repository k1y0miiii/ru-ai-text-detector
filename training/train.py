"""
Обучение классификатора поверх фич.

Берём НЕ нейросеть, а градиентный бустинг — потому что:
  - фич немного (8 штук), глубокая сеть тут избыточна и переобучится
  - бустинг даёт интерпретируемость: можно показать важность каждой фичи
  - модель весит килобайты, инференс мгновенный

Поверх бустинга — КАЛИБРОВКА (CalibratedClassifierCV): без неё вероятности
кучкуются у 0.2-0.8 и «0.7» не значит «70% шанс ИИ». Изотоническая калибровка
делает вероятности честными.

Это и есть тот "лёгкий детектор", про который ты слышал.
"""

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from aidetector.dataset import build_feature_matrix, load_jsonl
from aidetector.features import FEATURE_NAMES
from aidetector.paths import model_path


def _avg_importances(clf, X_tr, y_tr):
    """Важность фич из калиброванного ансамбля: усредняем по базовым бустингам,
    обученным на CV-фолдах. Если структура поменялась (версия sklearn) — фолбэк:
    обучаем отдельный бустинг на тех же данных только ради важностей."""
    imps = []
    for cc in getattr(clf, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None and hasattr(est, "feature_importances_"):
            imps.append(est.feature_importances_)
    if imps:
        return np.mean(imps, axis=0)
    g = GradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
    return g.feature_importances_


def train(jsonl_path: str, model_out: str = model_path("model.joblib")):
    rows = load_jsonl(jsonl_path)
    print(f"Загружено примеров: {len(rows)}")

    X, y = build_feature_matrix(rows)
    if len(set(y.tolist())) < 2:
        raise SystemExit("Нужны оба класса: и человек (0), и ИИ (1).")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # База — градиентный бустинг; поверх — изотоническая калибровка на 5 фолдах.
    base = GradientBoostingClassifier(random_state=42)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("\n=== Качество на тесте ===")
    print(classification_report(y_te, preds, target_names=["человек", "ИИ"]))
    if len(set(y_te.tolist())) == 2:
        print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")

        # КРИТИЧНО (см. CLAUDE.md): доля ложных обвинений — человеческие тексты,
        # ошибочно помеченные как ИИ. Позитивный класс = ИИ (1).
        # confusion_matrix с labels=[0,1] -> [[TN, FP], [FN, TP]].
        tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        recall_ai = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"FPR (человек ошибочно как ИИ): {fpr:.3f}  "
              f"(FP={fp} из {fp + tn} человеческих текстов)")
        print(f"recall по ИИ (на тесте, тексты >=200 симв): {recall_ai:.3f}")
        print(f"Матрица ошибок: TN={tn} FP={fp} FN={fn} TP={tp}")

        # Проверка калибровки: «0.7 действительно ≈ 70%?»
        print("\n=== Калибровка (хотим: predicted ≈ observed) ===")
        frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=5, strategy="quantile")
        print(f"  {'предсказано':>12} | {'факт ИИ':>10}")
        for mp, fpos in zip(mean_pred, frac_pos):
            print(f"  {mp:>12.2f} | {fpos:>10.2f}")

    print("\n=== Важность фич ===")
    importances = _avg_importances(clf, X_tr, y_tr)
    for name, imp in sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda t: t[1], reverse=True,
    ):
        print(f"  {name:20s} {imp:.3f}")

    joblib.dump(clf, model_out)
    print(f"\nМодель сохранена в {model_out}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/demo.jsonl"
    train(path)
