"""
Phase 4: Explainable AI — SHAP-based score explanations.

Generates human-readable explanations for each applicant's merit score,
converting SHAP feature contributions into plain-language text.
"""

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier


# Human-readable feature names for explanations
FEATURE_LABELS = {
    "requested_amount": "Requested subsidy amount",
    "norm_amount": "Standard norm amount",
    "total_applications": "Number of past applications",
    "total_requested": "Total historical funding requested",
    "avg_requested": "Average request amount",
    "approval_rate": "Historical approval rate",
    "rejection_rate": "Historical rejection rate",
    "history_days": "Track record length (days)",
    "unique_directions": "Diversity of farming activities",
    "ndvi_score": "Land use verification (NDVI)",
    "credit_score": "Credit score",
    "env_compliance": "Environmental compliance",
    "active_lawsuits": "Active lawsuits",
    "region": "Region",
    "direction": "Subsidy direction (farming type)",
    "district": "District",
    "has_tax_debt": "Tax debt status",
}


def compute_shap_values(
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> shap.Explanation:
    """Compute SHAP values for the given feature matrix."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def explain_single_applicant(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    top_n: int = 5,
) -> list[dict]:
    """Generate human-readable explanations for a single prediction.

    Args:
        shap_values: SHAP values array for one sample.
        feature_names: List of feature names.
        feature_values: Actual feature values for this sample.
        top_n: Number of top contributing factors to include.

    Returns:
        List of dicts with 'feature', 'value', 'impact', 'direction', 'text'.
    """
    # Sort features by absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    explanations = []
    for idx in top_indices:
        fname = feature_names[idx]
        shap_val = float(shap_values[idx])
        fval = feature_values.iloc[idx]
        label = FEATURE_LABELS.get(fname, fname)

        direction = "positive" if shap_val > 0 else "negative"
        # Scale SHAP impact to approximate point contribution (0-100 scale)
        impact_points = abs(shap_val) * 25  # rough scaling factor

        text = _generate_explanation_text(label, fval, direction, impact_points)

        explanations.append({
            "feature": fname,
            "label": label,
            "value": fval,
            "shap_value": shap_val,
            "impact_points": round(impact_points, 1),
            "direction": direction,
            "text": text,
        })

    return explanations


def _generate_explanation_text(
    label: str, value, direction: str, impact: float
) -> str:
    """Convert a SHAP contribution into a human-readable sentence."""
    impact_str = f"{impact:.0f}"
    verb = "increased" if direction == "positive" else "decreased"

    # Special cases for more natural language
    if label == "Historical approval rate":
        pct = f"{float(value) * 100:.0f}%" if isinstance(value, (int, float)) else value
        return f"Score {verb} by ~{impact_str} pts: approval rate of {pct}"

    if label == "Tax debt status":
        status = "has tax debt" if str(value).lower() in ("true", "1", "yes") else "no tax debt"
        return f"Score {verb} by ~{impact_str} pts: applicant {status}"

    if label == "Active lawsuits":
        return f"Score {verb} by ~{impact_str} pts: {value} active lawsuit(s)"

    if label == "Land use verification (NDVI)":
        return f"Score {verb} by ~{impact_str} pts: NDVI land-use score of {value}"

    if label == "Credit score":
        return f"Score {verb} by ~{impact_str} pts: credit score of {value}"

    if label == "Environmental compliance":
        return f"Score {verb} by ~{impact_str} pts: environmental compliance at {value}"

    if label == "Number of past applications":
        return f"Score {verb} by ~{impact_str} pts: {value} previous application(s)"

    # Generic fallback
    return f"Score {verb} by ~{impact_str} pts due to {label.lower()} ({value})"


def explain_all_applicants(
    model: CatBoostClassifier,
    scored_applicants: pd.DataFrame,
    transactions: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    top_n: int = 5,
) -> dict[str, list[dict]]:
    """Generate explanations for all scored applicants.

    Uses one representative transaction per applicant (the latest one)
    to compute SHAP values.

    Returns:
        Dict mapping bin_iin -> list of explanation dicts.
    """
    merge_cols = [
        "bin_iin", "total_applications", "total_requested", "avg_requested",
        "approval_rate", "rejection_rate", "history_days", "unique_directions",
        "has_tax_debt", "active_lawsuits", "ndvi_score", "credit_score",
        "env_compliance",
    ]
    df = transactions.merge(
        scored_applicants[merge_cols], on="bin_iin", how="inner"
    )

    # Take the latest transaction per applicant as representative
    df = df.sort_values("date_received").groupby("bin_iin").tail(1).reset_index(drop=True)

    X = df[feature_cols].copy()
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)
    for col in feature_cols:
        if col not in cat_features:
            X[col] = X[col].fillna(0)

    # Compute SHAP
    shap_vals = compute_shap_values(model, X)

    # Handle multi-output SHAP (take class=1 for binary)
    values = shap_vals.values
    if values.ndim == 3:
        values = values[:, :, 1]

    explanations = {}
    for i, bin_iin in enumerate(df["bin_iin"].values):
        explanations[bin_iin] = explain_single_applicant(
            values[i],
            feature_cols,
            X.iloc[i],
            top_n=top_n,
        )

    return explanations


if __name__ == "__main__":
    from src.data_ingestion import load_and_process
    from src.enrichment import enrich_applicants
    from src.model import prepare_training_data, train_model, compute_merit_scores, ALL_FEATURES, CATEGORICAL_FEATURES

    print("Loading data...")
    transactions, applicants = load_and_process()
    enriched = enrich_applicants(applicants)

    print("Training model...")
    train_df = prepare_training_data(transactions, enriched)
    model, metrics = train_model(train_df, iterations=200)
    scores = compute_merit_scores(model, enriched, transactions)

    print("\nGenerating explanations for top 5 applicants...")
    explanations = explain_all_applicants(
        model, scores.head(50), transactions, ALL_FEATURES, CATEGORICAL_FEATURES
    )

    for bin_iin in list(explanations.keys())[:5]:
        score = scores.loc[scores["bin_iin"] == bin_iin, "merit_score"].values[0]
        print(f"\n--- Applicant {bin_iin} (Score: {score}) ---")
        for exp in explanations[bin_iin]:
            print(f"  {exp['text']}")
