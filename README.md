# AgroScore AI — Merit-based Subsidy Scoring Engine

Built for **Decentrathon 5.0 Hackathon** (Case 2: Agriculture / GovTech).

## Problem

In Kazakhstan, agricultural subsidies are distributed on a "first-come, first-served" basis. This approach ignores applicant reliability and efficiency, reducing the economic return of government funding. AgroScore AI replaces this with a **merit-based ranking system** that scores every applicant on a 0-100 scale with full explainability.

## How It Works

1. **Data Ingestion** — Dynamically downloads 36,000+ subsidy applications from the Kazakhstan Agriculture Subsidy Registry (subsidy.plem.kz).
2. **Enrichment** — Augments applicant profiles with simulated external data: tax debt status, judicial history, satellite-based land use (NDVI), credit scores, and environmental compliance.
3. **ML Scoring** — CatBoost classifier predicts execution probability; output is normalized to a 0-100 Merit Score.
4. **Explainable AI** — SHAP values are converted into human-readable explanations (e.g., "Score increased by 15 pts due to perfect tax compliance").
5. **Dashboard** — Streamlit app for government officials with ranked lists, "Why this score?" breakdowns, and Approve/Reject buttons.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd indrive_scoring_case

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Run the Dashboard

```bash
uv run streamlit run app.py
```

The app will:
1. Download the dataset from Google Sheets (no local CSV needed)
2. Train the CatBoost model (~30 seconds)
3. Generate SHAP explanations
4. Launch the interactive dashboard at http://localhost:8501

### Run Individual Modules

```bash
# Test data ingestion
uv run python -m src.data_ingestion

# Test enrichment
uv run python -m src.enrichment

# Train model and see metrics
uv run python -m src.model

# Generate SHAP explanations
uv run python -m src.explainer
```

## Project Structure

```
indrive_scoring_case/
├── app.py                  # Streamlit dashboard (Phase 5)
├── src/
│   ├── data_ingestion.py   # Data download & cleaning (Phase 1)
│   ├── enrichment.py       # Mock 360-degree enrichment (Phase 2)
│   ├── model.py            # CatBoost training & scoring (Phase 3)
│   └── explainer.py        # SHAP explanations (Phase 4)
├── ARCHITECTURE.md         # Technical architecture details
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## Tech Stack

- **Python 3.12** — Core language
- **Pandas** — Data processing
- **CatBoost** — Gradient boosting for categorical-heavy tabular data
- **SHAP** — Model interpretability
- **Streamlit** — Interactive dashboard
- **Plotly** — Visualizations

## Key Metrics

- **Dataset:** 36,651 transactions, 13,184 unique applicants
- **Model AUC-ROC:** 0.988
- **Model Accuracy:** 92.8%
- **Data source:** Fetched dynamically at runtime (no local CSV storage)

## Data Source

The dataset is fetched at runtime from the Kazakhstan subsidy registry export. No local CSV file is required. The URL points to a Google Sheets export of the official subsidy.plem.kz data.
