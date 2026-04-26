"""
random_forest_model.py
----------------------
Concrete Random Forest implementation of BaseModel.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Scikit-learn Random Forest bagging classifier.

    Default params are passed in from config; any key accepted by
    RandomForestClassifier can be included.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 150,
        "max_depth": 6,
        "random_state": 42,
    }

    def __init__(self, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="Random Forest", params=merged)
        self.build()

    def build(self) -> "RandomForestModel":
        self.estimator = RandomForestClassifier(**self.params)
        return self

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestModel":
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

    def get_estimator(self) -> RandomForestClassifier:
        return self.estimator