"""
base_model.py
-------------
BaseModel: abstract interface that every concrete model must implement.
Enforces a consistent API across XGBoost, Random Forest, and LightGBM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
)
from src.utils import get_logger


class BaseModel(ABC):
    """
    Abstract base class for all models in the comparison pipeline.

    Every subclass must implement:
        - build()   — instantiate the underlying estimator
        - fit()     — train on (X_train, y_train)
        - predict() — return class labels for X
        - predict_proba() — return probability array for X
        - get_estimator() — return the raw fitted estimator

    The evaluate() method is shared and must not be overridden.
    """

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params
        self.estimator = None
        self._is_fitted = False
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def build(self) -> BaseModel:
        """Instantiate the underlying sklearn-compatible estimator."""

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseModel:
        """Train the model. Must set self._is_fitted = True."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities shaped (n_samples, n_classes)."""

    @abstractmethod
    def get_estimator(self):
        """Return the raw fitted estimator object."""

    # ------------------------------------------------------------------
    # Shared concrete methods
    # ------------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Compute Accuracy, ROC-AUC, and F1-score.

        Returns
        -------
        dict with keys: accuracy, roc_auc, f1, report
        """
        self._check_fitted()
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "report": classification_report(y_test, y_pred),
        }
        self.logger.info(
            f"{self.name} — Accuracy: {metrics['accuracy']} | "
            f"AUC: {metrics['roc_auc']} | F1: {metrics['f1']}"
        )
        return metrics

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"Model '{self.name}' has not been fitted yet. Call fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"