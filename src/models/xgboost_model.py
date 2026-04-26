"""
xgboost_model.py
----------------
Concrete XGBoost implementation of BaseModel.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost gradient boosting classifier.

    Default params are passed in from config; any key accepted by
    XGBClassifier can be included.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    def __init__(self, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="XGBoost", params=merged)
        self.build()

    def build(self) -> "XGBoostModel":
        self.estimator = XGBClassifier(**self.params)
        return self

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostModel":
        self.logger.info(f"Training {self.name}...")
        self.estimator.fit(X_train, y_train)
        self._is_fitted = True
        self.logger.info(f"{self.name} training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.estimator.predict_proba(X)

    def get_estimator(self) -> XGBClassifier:
        return self.estimator