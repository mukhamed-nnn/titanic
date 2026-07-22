"""Точка входа: загрузка данных → предобработка → обучение с CV → сабмит + метрики."""
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import config
from src.data import load_raw_data, preprocessing
from src.ensemble import average_ensemble, build_test_meta_features, fit_stacking
from src.models import NEEDS_SCALING, build_models
from src.training import run_model


def main():
    seed = config.general.seed
    cv = StratifiedKFold(n_splits=config.cv.n_splits, shuffle=config.cv.shuffle, random_state=seed)

    train_raw, test_raw = load_raw_data(config.paths.path_to_train, config.paths.path_to_test)
    df = pd.concat([train_raw, test_raw], sort=False).reset_index(drop=True)
    train, test, train_cat, test_cat = preprocessing(df)

    X, y = train.drop(columns=["Survived"]), train["Survived"]
    X_cat, y_cat = train_cat.drop(columns=["Survived"]), train_cat["Survived"]

    models = build_models(config, seed)

    stats, results = {}, {}
    for name, model in models.items():
        X_input, y_input = (X_cat, y_cat) if name == "cb" else (X, y)
        fit_params = {"cat_features": list(config.training.catboost.cat_features)} if name == "cb" else None
        results[name] = run_model(
            name, model, X_input, y_input, cv,
            needs_scaling=name in NEEDS_SCALING, fit_params=fit_params, stats=stats,
        )

    stacking_members = list(config.training.stacking.meta_members)
    stats["Averaging"] = average_ensemble(results, stacking_members, y)

    meta_model, meta_cv_score = fit_stacking(results, y, cv, config.training.stacking)
    stats["Stacking"] = meta_cv_score

    X_meta_test = build_test_meta_features(results, test, test_cat, stacking_members)
    final_preds = (meta_model.predict_proba(X_meta_test)[:, 1] >= 0.5).astype(int)

    submission = pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Survived": final_preds})
    submission.to_csv(config.paths.path_to_submission, index=False)

    with open(config.paths.path_to_metrics, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nМетрики (honest OOF / CV accuracy):")
    for name, acc in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<22} {acc:.4f}")
    print(f"\nСабмит сохранён в {config.paths.path_to_submission}")


if __name__ == "__main__":
    main()