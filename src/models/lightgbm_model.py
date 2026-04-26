"""
lightgbm_model.py
-----------------
Concrete LightGBM implementation of BaseModel.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from src.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM leaf-wise gradient boosting classifier.

    Default params are passed in from config; any key accepted by
    LGBMClassifier can be included.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(self, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="LightGBM", params=merged)
        self.build()

    def build(self) -> "LightGBMModel":
        self.estimator = LGBMClassifier(**self.params)
        return self

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMModel":
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

    def get_estimator(self) -> LGBMClassifier:
        return self.estimator