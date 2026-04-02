"""
Phase 3: CatBoost Merit Scoring Model.

The Merit Score is a composite of two independent signals:
  1. ML Probability (40%) — CatBoost predicts P(Executed) from application data.
  2. Enrichment Score (60%) — Weighted combination of livestock trends,
     financial health, legal standing, and operational metrics.

This avoids the "everyone gets 100" problem caused by the ML model alone
(most applicants have 100% historical approval, so the model gives them
near-identical high probabilities).

The enrichment score differentiates applicants by farm growth trajectory,
financial stability, and compliance — signals independent of past approval.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Features for CatBoost classifier
NUMERIC_FEATURES = [
    "requested_amount",
    "norm_amount",
    "total_applications",
    "total_requested",
    "avg_requested",
    "approval_rate",
    "rejection_rate",
    "history_days",
    "unique_directions",
    "ndvi_score",
    "credit_score",
    "env_compliance",
    "active_lawsuits",
    "large_livestock_latest",
    "small_livestock_latest",
    "milk_production_latest",
    "livestock_large_trend",
    "livestock_small_trend",
    "milk_trend",
    "annual_revenue",
]

CATEGORICAL_FEATURES = [
    "region",
    "direction",
    "district",
    "has_tax_debt",
    "is_bankrupt",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Enrichment columns needed for merge
ENRICHMENT_MERGE_COLS = [
    "bin_iin", "total_applications", "total_requested", "avg_requested",
    "approval_rate", "rejection_rate", "history_days", "unique_directions",
    "has_tax_debt", "active_lawsuits", "ndvi_score", "credit_score",
    "env_compliance", "is_bankrupt", "annual_revenue",
    "large_livestock_latest", "small_livestock_latest", "milk_production_latest",
    "livestock_large_trend", "livestock_small_trend", "milk_trend",
]


def prepare_training_data(
    transactions: pd.DataFrame,
    enriched_applicants: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-transaction data with applicant-level enrichment features.

    Only keeps rows with Executed or Rejected status for binary classification.
    """
    binary_mask = transactions["status_en"].isin(["Executed", "Rejected"])
    df = transactions[binary_mask].copy()
    df = df.merge(enriched_applicants[ENRICHMENT_MERGE_COLS], on="bin_iin", how="left")
    df["target"] = (df["status_en"] == "Executed").astype(int)
    return df


def train_model(
    train_df: pd.DataFrame,
    iterations: int = 500,
    random_seed: int = 42,
) -> tuple[CatBoostClassifier, dict]:
    """Train the CatBoost classifier and return model + metrics."""
    X = train_df[ALL_FEATURES].copy()
    y = train_df["target"]

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        X[col] = X[col].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    cat_feature_indices = [ALL_FEATURES.index(f) for f in CATEGORICAL_FEATURES]

    model = CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=50,
        cat_features=cat_feature_indices,
        auto_class_weights="Balanced",
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "auc_roc": auc,
        "accuracy": report["accuracy"],
        "precision_executed": report["1"]["precision"],
        "recall_executed": report["1"]["recall"],
        "f1_executed": report["1"]["f1-score"],
    }

    return model, metrics


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Rank values as percentiles (0-1). Handles NaN by assigning 0.5."""
    return series.rank(pct=True, method="average").fillna(0.5)


def compute_enrichment_score(enriched: pd.DataFrame) -> pd.Series:
    """Compute a 0-100 enrichment score from farm-level features.

    Weighted components:
    - Livestock growth trend (20%): Are herds growing or shrinking?
    - Farm scale (15%): Total livestock and milk production
    - Financial health (25%): Revenue, credit score, no bankruptcy/tax debt
    - Compliance (15%): NDVI, environmental compliance
    - Legal standing (10%): No lawsuits
    - Track record (15%): Approval rate, history length
    """
    df = enriched.copy()

    # --- Livestock growth (20%) ---
    # Average of large/small livestock and milk trends, mapped from [-1,1] to [0,1]
    trend_avg = (
        df["livestock_large_trend"].fillna(0)
        + df["livestock_small_trend"].fillna(0)
        + df["milk_trend"].fillna(0)
    ) / 3
    livestock_growth = ((trend_avg + 1) / 2).clip(0, 1)

    # --- Farm scale (15%) ---
    scale_large = _percentile_rank(df["large_livestock_latest"])
    scale_small = _percentile_rank(df["small_livestock_latest"])
    scale_milk = _percentile_rank(df["milk_production_latest"])
    farm_scale = (scale_large * 0.35 + scale_small * 0.35 + scale_milk * 0.3)

    # --- Financial health (25%) ---
    revenue_rank = _percentile_rank(df["annual_revenue"])
    credit_norm = ((df["credit_score"].fillna(650) - 300) / 550).clip(0, 1)
    no_bankruptcy = (~df["is_bankrupt"].fillna(False)).astype(float)
    no_tax_debt = (~df["has_tax_debt"].fillna(False)).astype(float)
    financial = (
        revenue_rank * 0.35
        + credit_norm * 0.25
        + no_bankruptcy * 0.25
        + no_tax_debt * 0.15
    )

    # --- Compliance (15%) ---
    ndvi = df["ndvi_score"].fillna(0.5)
    env = df["env_compliance"].fillna(0.5)
    compliance = ndvi * 0.5 + env * 0.5

    # --- Legal standing (10%) ---
    # 0 lawsuits = 1.0, 1 = 0.6, 2 = 0.3, 3+ = 0.0
    lawsuits = df["active_lawsuits"].fillna(0)
    legal = (1.0 - lawsuits / 3).clip(0, 1)

    # --- Track record (15%) ---
    approval = df["approval_rate"].fillna(0)
    history_rank = _percentile_rank(df["history_days"])
    track_record = approval * 0.6 + history_rank * 0.4

    # Weighted composite
    enrichment_score = (
        livestock_growth * 20
        + farm_scale * 15
        + financial * 25
        + compliance * 15
        + legal * 10
        + track_record * 15
    )

    return enrichment_score.round(1)


def compute_merit_scores(
    model: CatBoostClassifier,
    enriched_applicants: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute a composite Merit Score (0-100) for every unique applicant.

    Combines:
    - ML probability score (40%): CatBoost P(Executed), percentile-ranked
    - Enrichment score (60%): Farm growth, financial health, compliance
    """
    # Get ML predictions per transaction
    df = transactions.merge(
        enriched_applicants[ENRICHMENT_MERGE_COLS], on="bin_iin", how="left"
    )
    X = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        X[col] = X[col].fillna(0)

    proba = model.predict_proba(X)[:, 1]
    df["raw_score"] = proba

    # Aggregate ML score per applicant
    ml_scores = df.groupby("bin_iin").agg(
        mean_proba=("raw_score", "mean"),
    ).reset_index()

    # Percentile-rank the ML probabilities (spreads the distribution)
    ml_scores["ml_percentile"] = _percentile_rank(ml_scores["mean_proba"]) * 100

    # Compute enrichment score
    enrichment_score = compute_enrichment_score(enriched_applicants)
    enriched_with_score = enriched_applicants.copy()
    enriched_with_score["enrichment_score"] = enrichment_score

    # Merge
    scores = ml_scores.merge(
        enriched_with_score, on="bin_iin", how="left"
    )

    # Composite: 40% ML + 60% enrichment
    scores["merit_score"] = (
        scores["ml_percentile"] * 0.4
        + scores["enrichment_score"] * 0.6
    ).round(1)

    return scores.sort_values("merit_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_ingestion import load_and_process
    from src.enrichment import enrich_applicants

    print("Loading data...")
    transactions, applicants = load_and_process()
    enriched = enrich_applicants(applicants)

    print("Preparing training data...")
    train_df = prepare_training_data(transactions, enriched)
    print(f"Training samples: {len(train_df)} "
          f"(Executed: {train_df['target'].sum()}, "
          f"Rejected: {(~train_df['target'].astype(bool)).sum()})")

    print("\nTraining CatBoost model...")
    model, metrics = train_model(train_df)
    print("\nModel Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nComputing merit scores...")
    scores = compute_merit_scores(model, enriched, transactions)

    print(f"\nScore distribution:")
    print(scores["merit_score"].describe())
    print(f"\nScore buckets:")
    for bucket in [90, 80, 70, 60, 50, 40, 30]:
        count = (scores["merit_score"] >= bucket).sum()
        print(f"  >= {bucket}: {count} ({count / len(scores) * 100:.1f}%)")

    print(f"\nTop 10:")
    print(scores[["bin_iin", "merit_score", "ml_percentile", "enrichment_score",
                   "approval_rate", "livestock_large_trend", "is_bankrupt"]].head(10))
