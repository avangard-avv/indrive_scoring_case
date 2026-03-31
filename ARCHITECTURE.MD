# Architecture — AgroScore AI

## System Overview

AgroScore AI is a merit-based scoring engine that replaces "first-come, first-served" subsidy allocation with data-driven applicant ranking. The system processes Kazakhstan's agricultural subsidy registry, enriches it with external signals, trains a ML model, and explains every score.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data        │───>│  Enrichment  │───>│  CatBoost    │───>│  SHAP        │───>│  Streamlit   │
│  Ingestion   │    │  Module      │    │  Model       │    │  Explainer   │    │  Dashboard   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  36k+ rows           5 synthetic         Binary              Human-readable      Interactive
  from subsidy.       features per        classifier          text per             review UI
  plem.kz             applicant           -> 0-100 score      applicant
```

## Layer 1: Data Ingestion (`src/data_ingestion.py`)

### Download
- Fetches CSV dynamically from Google Sheets export URL
- Uses `requests` + `io.BytesIO` to handle UTF-8 Cyrillic encoding correctly
- Raw file has 4 header/metadata rows that are stripped during parsing

### Cleaning
- **Currency fields** (`Причитающая сумма`, `Норматив`): Removes non-breaking spaces (`\xa0`) used as thousand separators and converts Kazakh comma-decimal format to float
- **Dates** (`Дата поступления`): Parsed from `DD.MM.YYYY HH:MM:SS` format
- **Status mapping**: Russian statuses mapped to English (Исполнена -> Executed, Отклонена -> Rejected, etc.)
- **BIN/IIN extraction**: First 12 digits of `Номер заявки` identify the unique business entity

### Applicant Aggregation
Groups 36k+ transactions into ~13k unique applicants with features:
- `approval_rate` — Historical success ratio
- `total_requested` — Cumulative funding volume
- `history_days` — Track record length
- `unique_directions` — Diversification across farming types

## Layer 2: Data Enrichment (`src/enrichment.py`)

Simulates external data sources to build a "360-degree view" of each farmer. All enrichment is **deterministic** — same BIN/IIN always produces the same values (MD5 hash of BIN used as random seed).

| Feature | Source (Simulated) | Distribution | Rationale |
|---------|-------------------|--------------|-----------|
| `has_tax_debt` | Tax Authority (KGD) | ~15% prevalence | Tax compliance signals financial responsibility |
| `active_lawsuits` | Judicial system | 80% zero, Poisson-like | Legal disputes indicate operational risk |
| `ndvi_score` | Satellite imagery | Beta(5,2), skewed high | Land utilization verification |
| `credit_score` | Credit bureau | Normal(650, 80), clipped 300-850 | Financial health indicator |
| `env_compliance` | Environmental agency | Beta(7,2), skewed high | Regulatory compliance |

### Production Path
In production, these would be replaced with real API calls to:
- `data.egov.kz` — Government open data portal
- KGD tax authority API
- Unified judicial information system
- Kazakhstan Gharysh Sapary (satellite data)

## Layer 3: ML Model (`src/model.py`)

### Why CatBoost?
- Native categorical feature support (Region, Direction, District — all Cyrillic categories)
- Handles missing values natively
- Resistant to overfitting with auto class balancing

### Training Setup
- **Target**: Binary — Executed (1) vs Rejected (0)
- **Class balance**: ~21k Executed vs ~3k Rejected — handled via `auto_class_weights="Balanced"`
- **Features**: 13 numeric + 4 categorical
- **Split**: 80/20 stratified
- **Early stopping**: 50 rounds patience

### Merit Score Computation
1. Model predicts P(Executed) for every transaction
2. Per-applicant: average P(Executed) across all their transactions
3. Min-max normalize to 0-100 scale

### Performance
- AUC-ROC: **0.988**
- Accuracy: **92.8%**
- Precision (Executed): **99.4%**
- Recall (Executed): **92.4%**

## Layer 4: Explainable AI (`src/explainer.py`)

### SHAP Integration
- Uses `TreeExplainer` (exact, fast for tree models)
- One representative transaction per applicant (latest)
- Top-5 contributing features extracted per applicant

### Explanation Generation
SHAP values are converted to natural language:
```
"Score increased by ~40 pts due to historical rejection rate (0.0)"
"Score decreased by ~9 pts due to requested subsidy amount (6,564,480 ₸)"
"Score increased by ~15 pts: approval rate of 100%"
```

Feature-specific templates handle special cases (tax debt, lawsuits, NDVI) for more natural phrasing.

## Layer 5: Dashboard (`app.py`)

### Caching Strategy
- `@st.cache_data` — Data download and processing (TTL: 1 hour)
- `@st.cache_resource` — Trained model (persists across reruns)
- `@st.cache_data` — SHAP explanations

### UI Tabs
1. **Overview** — KPIs, score distribution histogram, regional/direction breakdowns
2. **Applicant Rankings** — Filterable table with score range slider, region filter, BIN search
3. **Detailed Review** — Single-applicant deep dive with SHAP waterfall chart, text explanations, and Approve/Reject/Defer buttons (human-in-the-loop)

### Column Configuration
Uses Streamlit's `column_config` for rich display: progress bars for scores, currency formatting for amounts, checkboxes for boolean flags.
