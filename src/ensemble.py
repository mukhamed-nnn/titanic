"""Усреднение и стекинг поверх честных OOF-вероятностей базовых моделей."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def average_ensemble(results: dict, members: list[str], y) -> float:
    """Простое усреднение OOF-вероятностей нескольких моделей — для сравнения со стекингом."""
    from sklearn.metrics import accuracy_score
    import numpy as np

    oof_avg = np.column_stack([results[name]["oof_proba"] for name in members]).mean(axis=1)
    return accuracy_score(y, (oof_avg >= 0.5).astype(int))


def fit_stacking(results: dict, y, cv, stacking_cfg):
    """Обучает мета-модель (LogisticRegression) на честных OOF-вероятностях базовых моделей.

    Возвращает (обученная мета-модель, честная CV-точность самой мета-модели).
    """
    members = list(stacking_cfg.meta_members)
    X_meta_train = pd.DataFrame({name: results[name]["oof_proba"] for name in members})

    meta_model = LogisticRegression(max_iter=1000, C=stacking_cfg.C, penalty=stacking_cfg.penalty)
    meta_cv_scores = cross_val_score(meta_model, X_meta_train, y, cv=cv, scoring="accuracy")
    meta_model.fit(X_meta_train, y)

    return meta_model, float(meta_cv_scores.mean())


def build_test_meta_features(results: dict, test, test_cat, members: list[str]) -> pd.DataFrame:
    """Строит матрицу мета-признаков для test — вероятности базовых моделей на новых данных."""
    probs = {}
    for name in members:
        model = results[name]["model"]
        scaler = results[name]["scaler"]
        X_input = test_cat if name == "cb" else test
        if scaler is not None:
            X_input = scaler.transform(X_input)
        probs[name] = model.predict_proba(X_input)[:, 1]
    return pd.DataFrame(probs)