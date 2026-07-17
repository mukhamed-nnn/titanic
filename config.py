from omegaconf import OmegaConf

config = {
    "general": {
        "experiment_name": "1",
        "seed": 11,
    },
    "paths": {
        "path_to_train": "./data/train.csv",
        "path_to_test": "./data/test.csv",
    },
    "models": {
        "logreg": {},
        "knn": {},
        "random_forest": {},
        "catboost": {},
        "lightgbm": {},
        "xgboost": {},
    },
}

config = OmegaConf.create(config)
