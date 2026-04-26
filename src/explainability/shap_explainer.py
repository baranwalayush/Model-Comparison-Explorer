"""
shap_explainer.py
-----------------
SHAPExplainer: generates and caches SHAP values for every registered model.

Uses shap.TreeExplainer (fast, exact) for all tree-based models.
Falls back to shap.KernelExplainer for any non-tree model added later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from dataclasses import dataclass, field
from typing import Optional
from src.models.model_registry import ModelRegistry
from src.utils import get_logger


@dataclass
class ExplainerResult:
    """
    Holds everything produced by SHAPExplainer for a single model.

    Attributes
    ----------
    model_name : str
    shap_values : np.ndarray           shape (n_samples, n_features)
    expected_value : float             base/expected value
    feature_names : list[str]
    X_test : pd.DataFrame              the test slice used
    shap_interaction : np.ndarray | None  optional interaction matrix
    """

    model_name: str
    shap_values: np.ndarray
    expected_value: float
    feature_names: list[str]
    X_test: pd.DataFrame
    shap_interaction: Optional[np.ndarray] = field(default=None, repr=False)

    def mean_abs_shap(self) -> pd.Series:
        """Mean |SHAP value| per feature — used for importance ranking."""
        return pd.Series(
            np.abs(self.shap_values).mean(axis=0),
            index=self.feature_names,
        ).sort_values(ascending=False)


class SHAPExplainer:
    """
    Computes SHAP explanations for all models in a ModelRegistry.

    Parameters
    ----------
    registry : ModelRegistry
    background_sample : int
        Rows sampled from X_train for shap background data.
    """

    def __init__(self, registry: ModelRegistry, background_sample: int = 100):
        self.registry = registry
        self.background_sample = background_sample
        self._results: dict[str, ExplainerResult] = {}
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        compute_interactions: bool = False,
    ) -> "SHAPExplainer":
        """
        Run SHAP for every model in the registry.

        Parameters
        ----------
        X_train : pd.DataFrame   Used as background data for the explainer.
        X_test  : pd.DataFrame   The samples to explain.
        compute_interactions : bool
            If True, also compute SHAP interaction values (slow for large data).
        """
        background = shap.sample(X_train, min(self.background_sample, len(X_train)))

        for name, model in self.registry:
            self.logger.info(f"Computing SHAP values for {name}...")
            result = self._explain_single(
                name=name,
                model=model,
                background=background,
                X_test=X_test,
                compute_interactions=compute_interactions,
            )
            self._results[name] = result
            self.logger.info(f"SHAP complete for {name}.")

        return self

    def get_result(self, model_name: str) -> ExplainerResult:
        if model_name not in self._results:
            raise KeyError(f"No SHAP results for '{model_name}'. Run explain_all() first.")
        return self._results[model_name]

    def get_all_results(self) -> dict[str, ExplainerResult]:
        return self._results

    def importance_comparison_df(self) -> pd.DataFrame:
        """
        Build a DataFrame with mean |SHAP value| per feature per model.
        Useful for side-by-side bar charts.

        Returns
        -------
        pd.DataFrame  shape (n_features, n_models)
        """
        return pd.DataFrame(
            {name: result.mean_abs_shap() for name, result in self._results.items()}
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _explain_single(
        self,
        name: str,
        model,
        background: pd.DataFrame,
        X_test: pd.DataFrame,
        compute_interactions: bool,
    ) -> ExplainerResult:
        estimator = model.get_estimator()
        feature_names = X_test.columns.tolist()

        # Convert data to numpy arrays to avoid dtype issues with SHAP
        X_test_np = X_test.values.astype(np.float64)
        background_np = background.values.astype(np.float64)

        try:
            explainer = shap.TreeExplainer(estimator, data=background_np)
            
            raw = explainer.shap_values(X_test_np, check_additivity=False)
        except Exception as e:
            self.logger.warning(
                f"TreeExplainer failed for {name} ({e}). Falling back to KernelExplainer."
            )
            # Use model.predict to avoid the list/indexing headache of predict_proba
            # unless you specifically need probabilities.
            # Wrap predict fn in a lambda to avoid SHAP trying to set feature_names_in_
            predict_fn = lambda x: estimator.predict(x)
            explainer = shap.KernelExplainer(predict_fn, background_np)
            raw = explainer.shap_values(X_test_np)

        # Handle the SHAP output format (differs between Tree and Kernel)
        # TreeExplainer for binary classification often returns a list [val_0, val_1]
        if isinstance(raw, list):
            # Check for binary classification vs regression
            shap_values = raw[1] if len(raw) > 1 else raw[0]
        elif isinstance(raw, np.ndarray) and len(raw.shape) == 3:
            # Some versions return (samples, features, classes)
            shap_values = raw[:, :, 1]
        else:
            shap_values = raw

        # Handle Expected Value logic
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            expected_value = float(ev[1] if len(ev) == 2 else ev[0])
        else:
            expected_value = float(ev)

        interaction = None
        if compute_interactions:
            try:
                interaction = explainer.shap_interaction_values(X_test_np, check_additivity=False)
                if isinstance(interaction, list):
                    interaction = interaction[1] if len(interaction) == 2 else interaction[0]
            except Exception as e:
                self.logger.warning(f"Interaction values failed for {name}: {e}")

        # try:
        #     explainer = shap.TreeExplainer(estimator, data=background)
        #     raw = explainer.shap_values(X_test)
        # except Exception as e:
        #     self.logger.warning(
        #         f"TreeExplainer failed for {name} ({e}). Falling back to KernelExplainer."
        #     )
        #     explainer = shap.KernelExplainer(model.predict_proba, background)
        #     raw = explainer.shap_values(X_test)

        # # For binary classifiers shap_values returns [neg_class, pos_class]
        # shap_values = raw[1] if isinstance(raw, list) else raw
        # expected_value = (
        #     explainer.expected_value[1]
        #     if isinstance(explainer.expected_value, (list, np.ndarray))
        #     else float(explainer.expected_value)
        # )

        # interaction = None
        # if compute_interactions:
        #     try:
        #         interaction = explainer.shap_interaction_values(X_test)
        #         if isinstance(interaction, list):
        #             interaction = interaction[1]
        #     except Exception as e:
        #         self.logger.warning(f"Interaction values failed for {name}: {e}")

        return ExplainerResult(
            model_name=name,
            shap_values=shap_values,
            expected_value=expected_value,
            feature_names=feature_names,
            X_test=X_test,
            shap_interaction=interaction,
        )