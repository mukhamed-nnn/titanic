import numpy as np
import torch
from omegaconf import OmegaConf

config = {
    "general": {
        "experiment_name": "titanic_baseline_v1",
        "seed": 0xDEF,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "paths": {
        "path_to_train": "./data/train.csv",
        "path_to_test": "./data/test.csv",
        "path_to_submission": "./output/submission.csv",
        "path_to_metrics": "./output/metrics.json",
    },
    "cv": {
        "n_splits": 5,
        "shuffle": True,
    },
    "preprocessing": {
        "cat_features": ["Embarked", "Title", "Deck"],
    },
    "training": {
        "dummy": {
            "strategy": "most_frequent",
        },
        "logreg": {
            "C": 0.023101,
            "solver": "lbfgs",
            "max_iter": 2000,
        },
        "knn": {
            "n_neighbors": 6,       
            "weights": "uniform",
            "metric": "manhattan",
            "p": 2,
        },
        "decision_tree": {
            "criterion": "log_loss",
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 13,
            "max_features": None,
        },
        "random_forest": {
            "n_estimators": 300,
            "criterion": "gini",
            "max_samples": 0.7,
            "max_depth": 5,
            "max_features": 0.6,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "catboost": {
            "depth": 5,
            "iterations": 300,
            "loss_function": "Logloss",
            "learning_rate": 0.05,
            "l2_leaf_reg": 5,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "cat_features": ["Embarked", "Title", "Deck"],
        },
        "lightgbm": {
            "n_estimators": 50,
            "learning_rate": 0.0775,
            "num_leaves": 8,
            "max_depth": 7,
            "min_child_samples": 10,
        },
        "xgboost": {
            "n_estimators": 50,
            "learning_rate": 0.0775,
            "max_depth": 7,
            "min_child_weight": 2,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
        },
        "nn": {
            "hidden_dim": 32,
            "dropout1": 0.3,
            "dropout2": 0.2,
            "num_epochs": 150,
            "lr": 0.005,
            "weight_decay": 1e-4,
            "early_stopping": {"patience": 10, "min_delta": 1e-4},
            "scheduler": {"patience": 10, "factor": 0.1},
        },
        "stacking": {
            "meta_members": ["lr", "tree", "rf", "lgbm", "xgb", "cb"],
            "C": 1.0,
        },
    },
    "dataloader": {"batch_size": 32, "num_workers": 2, "shuffle": True},
}

config = OmegaConf.create(config)
