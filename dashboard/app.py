"""
dashboard/app.py
----------------
Streamlit interactive dashboard for the SHAP Model Comparison Explorer.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.pipeline.comparison_pipeline import ModelComparisonPipeline


# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHAP Model Comparison Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────
# Pipeline (cached so it only runs once per session)
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Running pipeline — training models & computing SHAP values...")
def load_pipeline():
    pipeline = ModelComparisonPipeline(config_path="config/config.yaml")
    return pipeline.run()


artifacts = load_pipeline()
plotter   = artifacts.plotter
registry  = artifacts.registry
explainer = artifacts.explainer
metrics   = artifacts.metrics
model_names = registry.names


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Controls")
    st.markdown("---")

    selected_model = st.selectbox("Model", model_names)

    feature_names = artifacts.feature_names
    selected_feature = st.selectbox("Feature (dependence plot)", feature_names)

    sample_idx = st.slider(
        "Sample index (waterfall / force)",
        min_value=0,
        max_value=len(artifacts.X_test) - 1,
        value=0,
    )

    st.markdown("---")
    st.markdown("**Test set ground truth**")
    st.write(f"Survived: **{int(artifacts.y_test.iloc[sample_idx])}**")
    preds = {
        name: registry.get(name).predict(artifacts.X_test)[sample_idx]
        for name in model_names
    }
    for name, pred in preds.items():
        icon = "✅" if pred == artifacts.y_test.iloc[sample_idx] else "❌"
        st.write(f"{icon} {name}: **{pred}**")


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────
st.title("🔍 SHAP Model Comparison Explorer")
st.caption("Titanic Survival Prediction — XGBoost vs Random Forest vs LightGBM")
st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# Section 1: Model Metrics
# ──────────────────────────────────────────────────────────────────────
st.subheader("1 · Model Performance")
metric_cols = st.columns(len(model_names))
for col, name in zip(metric_cols, model_names):
    m = metrics[name]
    with col:
        st.metric(label=f"{name} — Accuracy", value=m["accuracy"])
        st.metric(label="ROC-AUC", value=m["roc_auc"])
        st.metric(label="F1 Score", value=m["f1"])

with st.expander("Full classification reports"):
    for name in model_names:
        st.markdown(f"**{name}**")
        st.code(metrics[name]["report"])


# ──────────────────────────────────────────────────────────────────────
# Section 2: Importance Comparison
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2 · Feature Importance Comparison (all models)")
st.caption("Mean |SHAP value| per feature. Larger = more influential.")

col_imp, col_heat = st.columns(2)
with col_imp:
    fig_imp = plotter.plot_importance_comparison(figsize=(8, 5))
    st.pyplot(fig_imp)
    plt.close(fig_imp)

with col_heat:
    fig_heat = plotter.plot_ranking_heatmap(figsize=(8, 5))
    st.pyplot(fig_heat)
    plt.close(fig_heat)


# ──────────────────────────────────────────────────────────────────────
# Section 3: Beeswarm / Summary Plot
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"3 · SHAP Beeswarm — {selected_model}")
st.caption(
    "Each dot is one passenger. Colour = feature value (red=high, blue=low). "
    "Position on x-axis = impact on prediction."
)
fig_bee = plotter.plot_beeswarm(selected_model, figsize=(10, 6))
st.pyplot(fig_bee)
plt.close(fig_bee)


# ──────────────────────────────────────────────────────────────────────
# Section 4: Waterfall — single prediction
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"4 · Waterfall Plot — Sample #{sample_idx}")
st.caption("How each feature nudges the prediction above or below the baseline.")

wf_cols = st.columns(len(model_names))
for col, name in zip(wf_cols, model_names):
    with col:
        st.markdown(f"**{name}**")
        fig_wf = plotter.plot_waterfall(name, sample_idx=sample_idx, figsize=(6, 5))
        st.pyplot(fig_wf)
        plt.close(fig_wf)


# ──────────────────────────────────────────────────────────────────────
# Section 5: Dependence Plot
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"5 · Dependence Plot — {selected_feature} | {selected_model}")
st.caption(
    "x-axis = raw feature value; y-axis = SHAP value; "
    "colour = most interacting feature (auto-selected)."
)
try:
    fig_dep = plotter.plot_dependence(selected_model, selected_feature, figsize=(9, 5))
    st.pyplot(fig_dep)
    plt.close(fig_dep)
except Exception as e:
    st.warning(f"Could not render dependence plot: {e}")


# ──────────────────────────────────────────────────────────────────────
# Section 6: Raw SHAP values table
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("6 · Raw SHAP values for this sample"):
    result = explainer.get_result(selected_model)
    shap_row = pd.Series(
        result.shap_values[sample_idx],
        index=result.feature_names,
    ).sort_values(key=abs, ascending=False)
    st.dataframe(
        shap_row.reset_index().rename(columns={"index": "Feature", 0: "SHAP value"}),
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with SHAP · XGBoost · LightGBM · Scikit-learn · Streamlit")