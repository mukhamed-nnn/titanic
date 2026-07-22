"""Честная OOF-оценка модели и построение финальной модели на 100% данных."""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def run_model(name, estimator, X, y, cv, needs_scaling=False, fit_params=None, stats=None):
    """Прогоняет модель через K-Fold, возвращает honest OOF accuracy и финальную модель.

    Работает одинаково для любой sklearn-совместимой модели (включая обёртки
    вокруг CatBoost/PyTorch), что позволяет использовать один и тот же код
    для всех архитектур в пайплайне.
    """
    fit_params = fit_params or {}
    oof_proba = np.zeros(len(y))
    fold_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_tr, X_val = scaler.fit_transform(X_tr), scaler.transform(X_val)

        model = clone(estimator)
        model.fit(X_tr, y_tr, **fit_params)
        fold_proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = fold_proba
        fold_scores.append(accuracy_score(y_val, (fold_proba >= 0.5).astype(int)))

    acc = accuracy_score(y, (oof_proba >= 0.5).astype(int))
    if stats is not None:
        stats[name] = acc
    print(f"{name:<5}: OOF accuracy = {acc:.4f} │ by folds: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    scaler_full = StandardScaler().fit(X) if needs_scaling else None
    X_full = scaler_full.transform(X) if scaler_full else X
    final_model = clone(estimator).fit(X_full, y, **fit_params)

    return {"oof_proba": oof_proba, "model": final_model, "scaler": scaler_full, "fold_scores": fold_scores}