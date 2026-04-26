# SHAP Model Comparison Explorer

A fully object-oriented ML project that trains **XGBoost**, **Random Forest**,
and **LightGBM** on the Titanic dataset, then uses **SHAP** to compare how each
model explains its predictions.

---

## Folder Structure

```
shap_comparator/
├── config/
│   └── config.yaml             # All tunable parameters live here
├── data/
│   ├── raw/                    # Drop custom CSV here (optional)
│   └── processed/
│       └── plots/              # Auto-generated plots (--save-plots)
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
├── notebooks/
│   └── exploration.ipynb       # (optional) EDA notebook
├── src/
│   ├── utils.py                # Config loader + logger
│   ├── data/
│   │   ├── loader.py           # DataLoader
│   │   └── preprocessor.py     # DataPreprocessor
│   ├── models/
│   │   ├── base_model.py       # Abstract BaseModel
│   │   ├── xgboost_model.py    # XGBoostModel
│   │   ├── random_forest_model.py
│   │   ├── lightgbm_model.py
│   │   └── model_registry.py   # ModelRegistry
│   ├── explainability/
│   │   └── shap_explainer.py   # SHAPExplainer + ExplainerResult
│   ├── visualization/
│   │   └── plotter.py          # SHAPPlotter
│   └── pipeline/
│       └── comparison_pipeline.py   # ModelComparisonPipeline (orchestrator)
├── main.py                     # CLI entry point
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the CLI pipeline
python main.py

# 4. (Optional) Save all plots as PNGs
python main.py --save-plots

# 5. Launch the interactive Streamlit dashboard
streamlit run dashboard/app.py
```

---

## Configuration

Edit `config/config.yaml` to change:
- Data source (`seaborn` or `csv`)
- Model hyperparameters
- SHAP display options
- Train/test split ratio

---

## Adding a New Model

1. Create `src/models/your_model.py`, subclass `BaseModel`, implement the 5 abstract methods.
2. Add it to `ModelRegistry._MODEL_MAP` in `model_registry.py`.
3. Add its config section in `config.yaml`.

That's it — the pipeline, explainer, and dashboard pick it up automatically.

---

## Class Hierarchy

```
BaseModel (ABC)
├── XGBoostModel
├── RandomForestModel
└── LightGBMModel

ModelRegistry          — stores and manages BaseModel instances
DataLoader             — fetches raw data
DataPreprocessor       — cleans, engineers, encodes, splits
SHAPExplainer          — wraps shap.TreeExplainer per model
  └── ExplainerResult  — dataclass holding shap_values + metadata
SHAPPlotter            — all matplotlib figures
ModelComparisonPipeline — orchestrates all of the above
  └── PipelineArtifacts — dataclass passed between stages & to dashboard
```