"""
Phase 3: CatBoost Merit Scoring Model.

Trains a CatBoost classifier on per-application data to predict
status (Executed vs Rejected), then uses predicted probabilities
to compute a normalized Merit Score (0-100).
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# Features used for the model
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
]

CATEGORICAL_FEATURES = [
    "region",
    "direction",
    "district",
    "has_tax_debt",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_training_data(
    transactions: pd.DataFrame,
    enriched_applicants: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-transaction data with applicant-level enrichment features.

    Only keeps rows with Executed or Rejected status for binary classification.
    """
    # Filter to binary outcomes only
    binary_mask = transactions["status_en"].isin(["Executed", "Rejected"])
    df = transactions[binary_mask].copy()

    # Merge enrichment data
    merge_cols = [
        "bin_iin", "total_applications", "total_requested", "avg_requested",
        "approval_rate", "rejection_rate", "history_days", "unique_directions",
        "has_tax_debt", "active_lawsuits", "ndvi_score", "credit_score",
        "env_compliance",
    ]
    df = df.merge(enriched_applicants[merge_cols], on="bin_iin", how="left")

    # Binary target: 1 = Executed, 0 = Rejected
    df["target"] = (df["status_en"] == "Executed").astype(int)

    return df


def train_model(
    train_df: pd.DataFrame,
    iterations: int = 500,
    random_seed: int = 42,
) -> tuple[CatBoostClassifier, dict]:
    """Train the CatBoost classifier and return model + metrics.

    Returns:
        (trained_model, metrics_dict)
    """
    # Prepare features
    X = train_df[ALL_FEATURES].copy()
    y = train_df["target"]

    # Fill NaN in categoricals
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)

    # Fill NaN in numerics
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

    # Evaluate
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


def compute_merit_scores(
    model: CatBoostClassifier,
    enriched_applicants: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute a normalized Merit Score (0-100) for every unique applicant.

    Uses the model's predicted probability of "Executed" status,
    averaged across all of an applicant's transactions, then scaled to 0-100.
    """
    # Get all transactions (not just Executed/Rejected) merged with enrichment
    merge_cols = [
        "bin_iin", "total_applications", "total_requested", "avg_requested",
        "approval_rate", "rejection_rate", "history_days", "unique_directions",
        "has_tax_debt", "active_lawsuits", "ndvi_score", "credit_score",
        "env_compliance",
    ]
    df = transactions.merge(enriched_applicants[merge_cols], on="bin_iin", how="left")

    X = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        X[col] = X[col].fillna(0)

    # Predict probability of "Executed"
    proba = model.predict_proba(X)[:, 1]
    df["raw_score"] = proba

    # Aggregate per applicant (mean probability across all their applications)
    scores = df.groupby("bin_iin").agg(
        mean_proba=("raw_score", "mean"),
    ).reset_index()

    # Normalize to 0-100 using min-max scaling
    min_p, max_p = scores["mean_proba"].min(), scores["mean_proba"].max()
    if max_p > min_p:
        scores["merit_score"] = (
            (scores["mean_proba"] - min_p) / (max_p - min_p) * 100
        ).round(1)
    else:
        scores["merit_score"] = 50.0

    # Merge back applicant info
    scores = scores.merge(enriched_applicants, on="bin_iin", how="left")

    return scores.sort_values("merit_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_ingestion import load_and_process
    from src.enrichment import enrich_applicants

    print("Loading data...")
    transactions, applicants = load_and_process()
    enriched = enrich_applicants(applicants)

    print("Preparing training data...")
    train_df = prepare_training_data(transactions, enriched)
    print(f"Training samples: {len(train_df)} (Executed: {train_df['target'].sum()}, Rejected: {(~train_df['target'].astype(bool)).sum()})")

    print("\nTraining CatBoost model...")
    model, metrics = train_model(train_df)
    print(f"\nModel Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nComputing merit scores...")
    scores = compute_merit_scores(model, enriched, transactions)
    print(f"\nTop 10 applicants by Merit Score:")
    print(scores[["bin_iin", "merit_score", "approval_rate", "total_applications", "primary_region"]].head(10))
