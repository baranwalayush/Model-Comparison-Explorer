"""
model_registry.py
-----------------
ModelRegistry: a central store that holds, trains, and evaluates
all registered BaseModel instances.

Adding a new model to the project requires only:
    1. Implement BaseModel
    2. registry.register(YourNewModel(params))
"""

from __future__ import annotations

import pandas as pd
from typing import Iterator
from src.models.base_model import BaseModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.lightgbm_model import LightGBMModel
from src.utils import get_logger


class ModelRegistry:
    """
    Manages the lifecycle of multiple BaseModel instances.

    Usage
    -----
    registry = ModelRegistry.from_config(config["models"])
    registry.fit_all(X_train, y_train)
    metrics = registry.evaluate_all(X_test, y_test)
    """

    # Map config keys → model classes 
    _MODEL_MAP = {
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
        "lightgbm": LightGBMModel,
    }

    def __init__(self):
        self._registry: dict[str, BaseModel] = {}
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, model_config: dict) -> "ModelRegistry":
        """
        Build a ModelRegistry from the 'models' section of config.yaml.

        Parameters
        ----------
        model_config : dict
            Example: {"xgboost": {"n_estimators": 100}, "lightgbm": {...}}
        """
        registry = cls()
        for key, params in model_config.items():
            if key not in cls._MODEL_MAP:
                registry.logger.warning(f"Unknown model key '{key}' — skipping.")
                continue
            model = cls._MODEL_MAP[key](params=params)
            registry.register(model)
        return registry

    # ------------------------------------------------------------------
    # Registry operations
    # ------------------------------------------------------------------

    def register(self, model: BaseModel) -> "ModelRegistry":
        """Add a model to the registry. Raises if name is already taken."""
        if model.name in self._registry:
            raise ValueError(f"A model named '{model.name}' is already registered.")
        self._registry[model.name] = model
        self.logger.info(f"Registered model: {model.name}")
        return self

    def get(self, name: str) -> BaseModel:
        """Retrieve a model by name."""
        if name not in self._registry:
            raise KeyError(f"No model named '{name}'. Available: {self.names}")
        return self._registry[name]

    def fit_all(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> "ModelRegistry":
        """Fit every registered model."""
        self.logger.info(f"Fitting {len(self._registry)} models...")
        for model in self._registry.values():
            model.fit(X_train, y_train)
        return self

    def evaluate_all(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, dict]:
        """
        Evaluate every registered model.

        Returns
        -------
        dict mapping model name → metrics dict
        """
        results = {}
        for name, model in self._registry.items():
            results[name] = model.evaluate(X_test, y_test)
        return results

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[str, BaseModel]]:
        return iter(self._registry.items())

    def __len__(self) -> int:
        return len(self._registry)

    @property
    def names(self) -> list[str]:
        return list(self._registry.keys())

    @property
    def models(self) -> list[BaseModel]:
        return list(self._registry.values())