# Architecture — AgroScore AI

## System Overview

AgroScore AI is a merit-based scoring engine that replaces "first-come, first-served" subsidy allocation with data-driven applicant ranking. The system processes Kazakhstan's agricultural subsidy registry, enriches it with livestock/financial/legal data, trains a ML model, and produces a composite Merit Score (0-100) with full explainability.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data        │───>│  Enrichment  │───>│  Composite   │───>│  SHAP +      │───>│  Streamlit   │
│  Ingestion   │    │  Module      │    │  Scoring     │    │  Breakdown   │    │  Dashboard   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  36k+ rows           Livestock,          CatBoost ML         Two-layer            Interactive
  from subsidy.       Revenue,            + Weighted          explanation:         review UI +
  plem.kz             Bankruptcy...       enrichment          SHAP + components    approve/reject
```

## Why Composite Scoring?

The original approach used only the ML model's P(Executed) as the score. **Problem:** the model achieved 0.988 AUC but 45.8% of applicants scored exactly 100/100 because ~87% of applications are executed. The ML model correctly learned that most applicants succeed, but it couldn't differentiate between good and excellent farmers.

**Solution:** A composite score combining two independent signals:
- **ML Probability (40%)** — CatBoost predicts execution likelihood (percentile-ranked to spread the distribution)
- **Enrichment Score (60%)** — Weighted farm performance metrics independent of historical approval

This produces a normal-like distribution: mean ~56, std ~14, range 22-89.

---

## Layer 1: Data Ingestion (`src/data_ingestion.py`)

### Download
- Fetches CSV dynamically from Google Sheets export URL at runtime
- Uses `requests` + `io.BytesIO` to handle UTF-8 Cyrillic encoding
- Raw file has 4 header/metadata rows that are stripped during parsing

### Cleaning
- **Currency fields** (`Причитающая сумма`, `Норматив`): Removes non-breaking spaces (`\xa0`) as thousand separators, converts Kazakh comma-decimal to float
- **Dates** (`Дата поступления`): Parsed from `DD.MM.YYYY HH:MM:SS`
- **Status mapping**: Russian -> English (Исполнена -> Executed, Отклонена -> Rejected, etc.)
- **BIN/IIN extraction**: First 12 digits of `Номер заявки` identify the unique business entity

### Applicant Aggregation
Groups 36k+ transactions into ~13k unique applicants with:
- `approval_rate`, `rejection_rate`
- `total_requested`, `avg_requested`
- `history_days`, `unique_directions`

---

## Layer 2: Data Enrichment (`src/enrichment.py`)

Simulates external data sources for a "360-degree view" of each farmer. All enrichment is **deterministic** (MD5 hash of BIN/IIN used as random seed — same applicant always gets the same values).

### Livestock Data (2021-2025)

| Feature | Description | Distribution |
|---------|-------------|--------------|
| `large_livestock_{year}` | Cows, horses count per year | LogNormal(3.5, 1.2), 5-2000 head |
| `small_livestock_{year}` | Sheep, goats count per year | LogNormal(4.5, 1.3), 10-15000 head |
| `milk_production_{year}` | Tons of milk per year | Correlated with large livestock |

**Year-over-year dynamics:**
- Large livestock: ~3% annual growth with 12% volatility
- Small livestock: ~4% annual growth with 15% volatility
- Milk production: Derived from cow count x dairy fraction x yield per head

**Trend computation:** Linear regression slope normalized by mean, clipped to [-1, 1]. Positive = growing farm, negative = shrinking.

### Financial & Legal

| Feature | Source (Simulated) | Distribution | Rationale |
|---------|-------------------|--------------|-----------|
| `annual_revenue` | Tax Authority | Correlated with herd size | Farm economic output |
| `is_bankrupt` | Court registry | ~5% prevalence | **Hard penalty** — bankrupt applicants should not receive subsidies |
| `has_tax_debt` | KGD Tax Authority | ~15% prevalence | Tax compliance indicates fiscal responsibility |
| `credit_score` | Credit bureau | Normal(650, 80), clipped 300-850 | Financial health and creditworthiness |

### Operational

| Feature | Source (Simulated) | Distribution | Rationale |
|---------|-------------------|--------------|-----------|
| `active_lawsuits` | Judicial system | 80% zero, 15% one, 5% two+ | Legal risk indicator |
| `ndvi_score` | Satellite imagery | Beta(5, 2) | Land utilization verification |
| `env_compliance` | Environmental agency | Beta(7, 2) | Regulatory compliance |

### Production Path
In production, these would connect to real APIs:
- `data.egov.kz` — Government open data (livestock statistics by region)
- KGD — Tax debt status
- Court registry — Bankruptcy and lawsuits
- Kazakhstan Gharysh Sapary — Satellite NDVI data

---

## Layer 3: Composite Scoring (`src/model.py`)

### CatBoost Classifier (40% weight)

**Why CatBoost?**
- Native categorical feature support (Region, Direction, District — all Cyrillic)
- 25 features: 20 numeric + 5 categorical (including livestock, financial, legal data)
- Auto class weight balancing (21k Executed vs 3k Rejected)

**Training:** 80/20 stratified split, early stopping at 50 rounds patience.

**Output:** P(Executed) per transaction, averaged per applicant, then **percentile-ranked** (not min-max) to spread the distribution evenly from 0-100.

### Enrichment Score (60% weight)

Six weighted components computed directly from enrichment features:

```
Merit Score = ML_percentile × 0.4 + Enrichment_score × 0.6

Enrichment Score (0-100) =
    Livestock Growth    × 20%   (herd trend direction over 2021-2025)
  + Farm Scale          × 15%   (total livestock + milk production)
  + Financial Health    × 25%   (revenue + credit + no bankruptcy + no tax debt)
  + Compliance          × 15%   (NDVI + environmental)
  + Legal Standing      × 10%   (no active lawsuits)
  + Track Record        × 15%   (approval rate + history length)
```

**Why 60/40 split?** The enrichment data differentiates applicants on dimensions the ML model can't distinguish (most have similar approval histories). The ML model adds value for edge cases where application-level patterns matter.

### Score Distribution
- Mean: ~56, Std: ~14
- Range: 22-89
- No artificial ceiling at 100

---

## Layer 4: Explainable AI (`src/explainer.py`)

### Two-layer Explanation

Every applicant gets **two types of explanation**:

**1. SHAP (ML component)**
- `TreeExplainer` for exact SHAP values
- Top-5 features per applicant
- Feature-specific natural language templates:
  - *"Score increased by ~34 pts due to historical rejection rate (0.0)"*
  - *"Score decreased by ~8 pts: applicant has tax debt"*
  - *"Score increased by ~7 pts: large herd is growing (+5.7%)"*

**2. Enrichment Breakdown (composite component)**
Shows all 6 weighted components with actual vs max scores:
```
🟢 Livestock Growth (10.4/20): Herds are growing (avg trend: +0.044)
🟢 Financial Health (21.9/25): Revenue ₸110M, credit 632
🔴 Legal Standing (6.7/10): 1 active lawsuit(s)
```

---

## Layer 5: Dashboard (`app.py`)

### Tabs

1. **Overview** — KPIs, score histogram, regional/direction breakdowns, bankruptcy/tax debt impact charts
2. **Applicant Rankings** — Filterable table (score range, region, BIN search, bankrupt-only filter) with livestock, financial, and legal columns
3. **Detailed Review** — Single-applicant deep dive:
   - Header metrics (merit score, ML score, enrichment score, revenue, credit, bankruptcy, lawsuits)
   - Livestock trend charts (2021-2025 line charts for large/small livestock, bar chart for milk)
   - Enrichment score breakdown (component bar chart + text)
   - SHAP waterfall chart + text explanations
   - Approve/Reject/Defer buttons (human-in-the-loop)

### Caching
- `@st.cache_data(ttl=3600)` — Data download
- `@st.cache_resource` — Trained model (persists across reruns)
- `@st.cache_data` — SHAP explanations
