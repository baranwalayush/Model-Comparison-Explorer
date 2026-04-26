"""
plotter.py
----------
SHAPPlotter: all SHAP visualisations in one place.

Every method returns a matplotlib Figure so it can be rendered
both in scripts (plt.show) and in Streamlit (st.pyplot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.explainability.shap_explainer import ExplainerResult, SHAPExplainer
from src.utils import get_logger

matplotlib.use("Agg")   # non-interactive backend; safe for Streamlit


class SHAPPlotter:
    """
    Wraps all SHAP plot types used in the comparison dashboard.

    Parameters
    ----------
    explainer : SHAPExplainer
        Must have explain_all() called before any plot method is used.
    max_display : int
        Maximum number of features shown in summary / bar plots.
    output_dir : str | None
        If provided, figures are also saved to this directory as PNGs.
    """

    def __init__(
        self,
        explainer: SHAPExplainer,
        max_display: int = 12,
        output_dir: Optional[str] = None,
    ):
        self.explainer = explainer
        self.max_display = max_display
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # 1. Side-by-side feature importance bar chart
    # ------------------------------------------------------------------

    def plot_importance_comparison(self, figsize: tuple = (12, 7)) -> plt.Figure:
        """
        Grouped horizontal bar chart: mean |SHAP| per feature per model.
        This is the flagship comparison view.
        """
        df = self.explainer.importance_comparison_df()
        df = df.head(self.max_display)

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(df))
        n_models = len(df.columns)
        bar_width = 0.22
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

        colors = ["#5B4FCF", "#E8593C", "#1A9E75"]
        for i, (model_name, color) in enumerate(zip(df.columns, colors)):
            ax.barh(
                x + offsets[i],
                df[model_name],
                height=bar_width,
                label=model_name,
                color=color,
                alpha=0.85,
            )

        ax.set_yticks(x)
        ax.set_yticklabels(df.index, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title("Feature Importance Comparison — All Models", fontsize=14, pad=14)
        ax.legend(fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._maybe_save(fig, "importance_comparison.png")
        return fig

    # ------------------------------------------------------------------
    # 2. SHAP summary (beeswarm) plot for a single model
    # ------------------------------------------------------------------

    def plot_beeswarm(self, model_name: str, figsize: tuple = (10, 7)) -> plt.Figure:
        """
        SHAP dot/beeswarm plot for one model — shows both direction
        and magnitude for every feature.
        """
        result = self.explainer.get_result(model_name)
        fig, ax = plt.subplots(figsize=figsize)
        shap.summary_plot(
            result.shap_values,
            result.X_test,
            max_display=self.max_display,
            show=False,
            plot_type="dot",
        )
        plt.title(f"SHAP Beeswarm — {model_name}", fontsize=13, pad=12)
        fig = plt.gcf()
        fig.tight_layout()
        self._maybe_save(fig, f"beeswarm_{model_name.replace(' ', '_')}.png")
        return fig

    # ------------------------------------------------------------------
    # 3. SHAP waterfall plot for a single prediction
    # ------------------------------------------------------------------

    def plot_waterfall(
        self, model_name: str, sample_idx: int = 0, figsize: tuple = (10, 6)
    ) -> plt.Figure:
        """
        Waterfall plot explaining a single prediction — shows how each
        feature pushes the output above or below the base value.
        """
        result = self.explainer.get_result(model_name)
        explanation = shap.Explanation(
            values=result.shap_values[sample_idx],
            base_values=result.expected_value,
            data=result.X_test.iloc[sample_idx].values,
            feature_names=result.feature_names,
        )
        fig, ax = plt.subplots(figsize=figsize)
        shap.waterfall_plot(explanation, max_display=self.max_display, show=False)
        plt.title(f"Waterfall — {model_name} | Sample #{sample_idx}", fontsize=13, pad=12)
        fig = plt.gcf()
        fig.tight_layout()
        self._maybe_save(fig, f"waterfall_{model_name.replace(' ', '_')}_{sample_idx}.png")
        return fig

    # ------------------------------------------------------------------
    # 4. SHAP dependence plot — feature vs SHAP value coloured by interaction
    # ------------------------------------------------------------------

    def plot_dependence(
        self,
        model_name: str,
        feature: str,
        interaction_feature: str = "auto",
        figsize: tuple = (9, 6),
    ) -> plt.Figure:
        """
        Dependence scatter plot: x-axis = raw feature value,
        y-axis = SHAP value, colour = a second feature for interaction insight.
        """
        result = self.explainer.get_result(model_name)
        fig, ax = plt.subplots(figsize=figsize)
        shap.dependence_plot(
            feature,
            result.shap_values,
            result.X_test,
            interaction_index=interaction_feature,
            ax=ax,
            show=False,
        )
        ax.set_title(
            f"Dependence — {feature} | {model_name}", fontsize=13, pad=12
        )
        fig.tight_layout()
        self._maybe_save(fig, f"dependence_{model_name.replace(' ','_')}_{feature}.png")
        return fig

    # ------------------------------------------------------------------
    # 5. SHAP heatmap for model agreement/disagreement
    # ------------------------------------------------------------------

    def plot_ranking_heatmap(self, top_n: int = 10, figsize: tuple = (10, 5)) -> plt.Figure:
        """
        Heatmap of feature *rank* per model — highlights where models
        agree or disagree on the most important features.
        """
        df = self.explainer.importance_comparison_df().head(top_n)
        rank_df = df.rank(ascending=False).astype(int)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.cm.RdYlGn_r
        im = ax.imshow(rank_df.T.values, aspect="auto", cmap=cmap, vmin=1, vmax=top_n)

        ax.set_xticks(range(len(rank_df)))
        ax.set_xticklabels(rank_df.index, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(len(rank_df.columns)))
        ax.set_yticklabels(rank_df.columns, fontsize=11)

        for i in range(len(rank_df.columns)):
            for j in range(len(rank_df)):
                ax.text(j, i, str(rank_df.values[j, i]),
                        ha="center", va="center", fontsize=10, fontweight="bold")

        fig.colorbar(im, ax=ax, label="Rank (1 = most important)")
        ax.set_title("Feature Importance Rank per Model", fontsize=13, pad=12)
        fig.tight_layout()
        self._maybe_save(fig, "ranking_heatmap.png")
        return fig

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _maybe_save(self, fig: plt.Figure, filename: str):
        if self.output_dir:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self.logger.info(f"Saved plot: {path}")