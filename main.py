"""
main.py
-------
CLI entry point.  Run the full pipeline, print metrics, and
optionally save all plots to data/processed/.

Usage:
    python main.py
    python main.py --config config/config.yaml --save-plots
"""

import argparse
import matplotlib.pyplot as plt
from src.pipeline.comparison_pipeline import ModelComparisonPipeline
from src.utils import get_logger

logger = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="SHAP Model Comparison Explorer")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save all plots to data/processed/plots/",
    )
    parser.add_argument(
        "--interactions",
        action="store_true",
        help="Also compute SHAP interaction values (slow)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = "data/processed/plots" if args.save_plots else None

    pipeline = ModelComparisonPipeline(
        config_path=args.config,
        output_dir=output_dir,
    )
    artifacts = pipeline.run(compute_interactions=args.interactions)

    plotter = artifacts.plotter
    model_names = artifacts.registry.names

    # -- Importance comparison (all models) --
    fig = plotter.plot_importance_comparison()
    plt.show()

    # -- Beeswarm for each model --
    for name in model_names:
        fig = plotter.plot_beeswarm(name)
        plt.show()

    # -- Waterfall for sample #0 per model --
    for name in model_names:
        fig = plotter.plot_waterfall(name, sample_idx=0)
        plt.show()

    # -- Ranking heatmap --
    fig = plotter.plot_ranking_heatmap()
    plt.show()

    logger.info("All plots rendered. Run `streamlit run dashboard/app.py` for the interactive dashboard.")


if __name__ == "__main__":
    main()