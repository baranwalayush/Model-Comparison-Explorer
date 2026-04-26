"""
comparison_pipeline.py
-----------------------
ModelComparisonPipeline: top-level orchestrator.

Wires together DataLoader → DataPreprocessor → ModelRegistry →
SHAPExplainer → SHAPPlotter in a single composable object.

This is the only class that Streamlit and main.py need to import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_registry import ModelRegistry
from src.explainability.shap_explainer import SHAPExplainer
from src.visualization.plotter import SHAPPlotter
from src.utils import get_logger, load_config


@dataclass
class PipelineArtifacts:
    """
    All artefacts produced by the pipeline, passed between stages
    and exposed to the dashboard.
    """

    X_train: pd.DataFrame = field(default=None, repr=False)
    X_test: pd.DataFrame = field(default=None, repr=False)
    y_train: pd.Series = field(default=None, repr=False)
    y_test: pd.Series = field(default=None, repr=False)
    metrics: dict = field(default_factory=dict)
    registry: Optional[ModelRegistry] = field(default=None, repr=False)
    explainer: Optional[SHAPExplainer] = field(default=None, repr=False)
    plotter: Optional[SHAPPlotter] = field(default=None, repr=False)
    feature_names: list = field(default_factory=list)


class ModelComparisonPipeline:
    """
    Orchestrates the full SHAP model comparison workflow.

    Parameters
    ----------
    config_path : str
        Path to config/config.yaml.
    output_dir : str | None
        If set, plots are saved here as PNGs.

    Usage
    -----
        pipeline = ModelComparisonPipeline("config/config.yaml")
        artifacts = pipeline.run()
        fig = artifacts.plotter.plot_importance_comparison()
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        output_dir: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.output_dir = output_dir
        self.artifacts = PipelineArtifacts()
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, compute_interactions: bool = False) -> PipelineArtifacts:
        """
        Execute the full pipeline end-to-end.

        Steps
        -----
        1. Load data
        2. Preprocess
        3. Train models
        4. Evaluate models
        5. Compute SHAP values
        6. Initialise plotter

        Returns
        -------
        PipelineArtifacts — everything the dashboard needs.
        """
        self.logger.info("=" * 55)
        self.logger.info("  SHAP Model Comparison Pipeline — START")
        self.logger.info("=" * 55)

        self._stage_load()
        self._stage_preprocess()
        self._stage_train_and_evaluate()
        self._stage_explain(compute_interactions)
        self._stage_plotter()

        self.logger.info("=" * 55)
        self.logger.info("  Pipeline complete.")
        self.logger.info("=" * 55)
        return self.artifacts

    # ------------------------------------------------------------------
    # Pipeline stages (each mutates self.artifacts in place)
    # ------------------------------------------------------------------

    def _stage_load(self):
        self.logger.info("[1/5] Loading data...")
        cfg = self.config["data"]
        loader = DataLoader(source=cfg["source"], csv_path=cfg.get("csv_path", ""))
        self.artifacts._raw_df = loader.load()

    def _stage_preprocess(self):
        self.logger.info("[2/5] Preprocessing...")
        cfg = self.config["data"]
        preprocessor = DataPreprocessor(
            target_column=cfg["target_column"],
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
        )
        (
            self.artifacts.X_train,
            self.artifacts.X_test,
            self.artifacts.y_train,
            self.artifacts.y_test,
        ) = preprocessor.fit_transform(self.artifacts._raw_df)
        self.artifacts.feature_names = preprocessor.get_feature_names()

    def _stage_train_and_evaluate(self):
        self.logger.info("[3/5] Training models...")
        registry = ModelRegistry.from_config(self.config["models"])
        registry.fit_all(self.artifacts.X_train, self.artifacts.y_train)

        self.logger.info("[4/5] Evaluating models...")
        self.artifacts.metrics = registry.evaluate_all(
            self.artifacts.X_test, self.artifacts.y_test
        )
        self.artifacts.registry = registry
        self._log_metrics_summary()

    def _stage_explain(self, compute_interactions: bool):
        self.logger.info("[5/5] Computing SHAP values...")
        shap_cfg = self.config.get("shap", {})
        explainer = SHAPExplainer(
            registry=self.artifacts.registry,
            background_sample=shap_cfg.get("background_sample", 100),
        )
        explainer.explain_all(
            X_train=self.artifacts.X_train,
            X_test=self.artifacts.X_test,
            compute_interactions=compute_interactions,
        )
        self.artifacts.explainer = explainer

    def _stage_plotter(self):
        shap_cfg = self.config.get("shap", {})
        self.artifacts.plotter = SHAPPlotter(
            explainer=self.artifacts.explainer,
            max_display=shap_cfg.get("max_display", 12),
            output_dir=self.output_dir,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_metrics_summary(self):
        self.logger.info("─" * 45)
        self.logger.info(f"{'Model':<20} {'Acc':>8} {'AUC':>8} {'F1':>8}")
        self.logger.info("─" * 45)
        for name, m in self.artifacts.metrics.items():
            self.logger.info(
                f"{name:<20} {m['accuracy']:>8} {m['roc_auc']:>8} {m['f1']:>8}"
            )
        self.logger.info("─" * 45)