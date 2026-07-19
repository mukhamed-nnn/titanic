import numpy as np
import torch
from omegaconf import OmegaConf

config = {
    "general": {
        "experiment_name": "1",
        "seed": 11,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "paths": {
        "path_to_train": "./data/train.csv",
        "path_to_test": "./data/test.csv",
        "path_to_submission": "./output_titanic.csv",
    },
    "training": {
        "logreg": {
            "penalty": "elasticnet",
            "C": np.float64(0.0657933224657568),
            "l1_ratio": 0.2,
            "solver": "saga",
            "max_iter": 2000,
        },
        "catboost": {
            "depth": 5,
            "iterations": 1500,
            "loss_function": "LogLoss",
            "early_stopping_rounds": 100,
            "cat_features": ["Name", "Embarked", "Deck", "Title"],
        },
        "nn": {
            "num_epochs": 150,
            "lr": 0.005,
            "weight_decay": 1e-4,
            "early_stopping": True,
            "scheduler": {"patience": 10, "factor": 0.1},
        },
        "stacking": {"n_splits": 10, "penalty": "l2", "C": 1.0},
    },
    "dataloader": {"batch_size": 32, "num_workers": 2, "shuffle": True},
}

config = OmegaConf.create(config)
