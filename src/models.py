"""Построение всех моделей пайплайна с гиперпараметрами, зафиксированными в config."""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# Модели, которым нужны отмасштабированные признаки
NEEDS_SCALING = {"lr", "knn", "nn"}


class NN3(nn.Module):
    """MLP: два скрытых слоя с BatchNorm и Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout1)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout2)
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer_1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.layer_2(x))
        x = self.dropout2(x)
        return self.output(x)


class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-совместимая обёртка над NN3: fit/predict_proba/clone работают как у любой sklearn-модели.

    Внутри себя разбивает train на train/val для early stopping — этот сплит
    независим от внешней кросс-валидации в run_model, поэтому снаружи модель
    не нуждается в needs_scaling=True: масштабирование признаков она делает сама.
    """

    def __init__(self, hidden_dim=32, dropout1=0.3, dropout2=0.2, lr=0.005, weight_decay=1e-4,
                 num_epochs=150, batch_size=32, val_split=0.2, patience=10, min_delta=1e-4,
                 scheduler_patience=10, scheduler_factor=0.1, seed=0, device=None):
        self.hidden_dim = hidden_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.patience = patience
        self.min_delta = min_delta
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.seed, stratify=y
        )

        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)
        X_val_scaled = self.scaler_.transform(X_val)

        self.model_ = NN3(X_train_scaled.shape[1], self.hidden_dim, self.dropout1, self.dropout2).to(self.device)

        class_counts = pd.Series(y_train).value_counts()
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.scheduler_patience, factor=self.scheduler_factor
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device),
                torch.tensor(y_train, dtype=torch.float32).to(self.device),
            ),
            batch_size=self.batch_size, shuffle=True,
        )
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for _ in range(self.num_epochs):
            self.model_.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch.view(-1, 1))
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_val_t), y_val_t.view(-1, 1)).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_weights is not None:
            self.model_.load_state_dict(best_weights)
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        with torch.no_grad():
            probs_1 = torch.sigmoid(self.model_(torch.tensor(X_scaled).to(self.device))).cpu().numpy().flatten()
        return np.column_stack([1 - probs_1, probs_1])


def build_models(cfg, seed: int) -> dict:
    """Возвращает словарь {название: необученная модель} по конфигу."""
    t = cfg.training
    return {
        "dummy": DummyClassifier(strategy=t.dummy.strategy),
        "lr": LogisticRegression(random_state=seed, **t.logreg),
        "knn": KNeighborsClassifier(**t.knn),
        "tree": DecisionTreeClassifier(random_state=seed, **t.decision_tree),
        "rf": RandomForestClassifier(random_state=seed, **t.random_forest),
        "lgbm": LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **t.lightgbm),
        "xgb": XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=-1, **t.xgboost),
        "cb": CatBoostClassifier(
            random_seed=seed, logging_level="Silent", allow_writing_files=False,
            **{k: v for k, v in t.catboost.items() if k != "cat_features"},
        ),
        "nn": TorchNNClassifier(
            seed=seed,
            hidden_dim=t.nn.hidden_dim, dropout1=t.nn.dropout1, dropout2=t.nn.dropout2,
            lr=t.nn.lr, weight_decay=t.nn.weight_decay, num_epochs=t.nn.num_epochs,
            patience=t.nn.early_stopping.patience, min_delta=t.nn.early_stopping.min_delta,
            scheduler_patience=t.nn.scheduler.patience, scheduler_factor=t.nn.scheduler.factor,
        ),
    }