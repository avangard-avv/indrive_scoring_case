"""
Phase 4: Explainable AI — SHAP + Enrichment score breakdown.

Generates human-readable explanations for each applicant's merit score.
Two layers of explanation:
1. SHAP values from CatBoost (explains the ML component)
2. Enrichment score breakdown (explains the composite component)
"""

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier

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
    "large_livestock_latest": "Large livestock count (2025)",
    "small_livestock_latest": "Small livestock count (2025)",
    "milk_production_latest": "Milk production (2025, tons)",
    "livestock_large_trend": "Large livestock growth trend",
    "livestock_small_trend": "Small livestock growth trend",
    "milk_trend": "Milk production trend",
    "annual_revenue": "Annual revenue (KZT)",
    "region": "Region",
    "direction": "Subsidy direction (farming type)",
    "district": "District",
    "has_tax_debt": "Tax debt status",
    "is_bankrupt": "Bankruptcy status",
}


def compute_shap_values(
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> shap.Explanation:
    """Compute SHAP values for the given feature matrix."""
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def explain_single_applicant(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    top_n: int = 5,
) -> list[dict]:
    """Generate human-readable explanations for a single ML prediction."""
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    explanations = []
    for idx in top_indices:
        fname = feature_names[idx]
        shap_val = float(shap_values[idx])
        fval = feature_values.iloc[idx]
        label = FEATURE_LABELS.get(fname, fname)
        direction = "positive" if shap_val > 0 else "negative"
        impact_points = abs(shap_val) * 25

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

    if label == "Historical approval rate":
        pct = f"{float(value) * 100:.0f}%" if isinstance(value, (int, float)) else value
        return f"Score {verb} by ~{impact_str} pts: approval rate of {pct}"

    if label == "Tax debt status":
        status = "has tax debt" if str(value).lower() in ("true", "1", "yes") else "no tax debt"
        return f"Score {verb} by ~{impact_str} pts: applicant {status}"

    if label == "Bankruptcy status":
        status = "is bankrupt" if str(value).lower() in ("true", "1", "yes") else "not bankrupt"
        return f"Score {verb} by ~{impact_str} pts: applicant {status}"

    if label == "Active lawsuits":
        return f"Score {verb} by ~{impact_str} pts: {value} active lawsuit(s)"

    if label == "Land use verification (NDVI)":
        return f"Score {verb} by ~{impact_str} pts: NDVI land-use score of {value}"

    if label == "Credit score":
        return f"Score {verb} by ~{impact_str} pts: credit score of {value}"

    if label == "Large livestock count (2025)":
        return f"Score {verb} by ~{impact_str} pts: {int(value)} large livestock (cows/horses)"

    if label == "Small livestock count (2025)":
        return f"Score {verb} by ~{impact_str} pts: {int(value)} small livestock (sheep/goats)"

    if label == "Milk production (2025, tons)":
        return f"Score {verb} by ~{impact_str} pts: {value:.1f} tons milk production"

    if label == "Large livestock growth trend":
        trend_word = "growing" if float(value) > 0 else "declining"
        return f"Score {verb} by ~{impact_str} pts: large herd is {trend_word} ({value:+.1%})"

    if label == "Small livestock growth trend":
        trend_word = "growing" if float(value) > 0 else "declining"
        return f"Score {verb} by ~{impact_str} pts: small herd is {trend_word} ({value:+.1%})"

    if label == "Milk production trend":
        trend_word = "increasing" if float(value) > 0 else "decreasing"
        return f"Score {verb} by ~{impact_str} pts: milk production {trend_word} ({value:+.1%})"

    if label == "Annual revenue (KZT)":
        return f"Score {verb} by ~{impact_str} pts: annual revenue ₸{float(value):,.0f}"

    if label == "Number of past applications":
        return f"Score {verb} by ~{impact_str} pts: {value} previous application(s)"

    return f"Score {verb} by ~{impact_str} pts due to {label.lower()} ({value})"


def generate_enrichment_breakdown(row: pd.Series) -> list[dict]:
    """Generate the enrichment score breakdown for one applicant.

    Shows the 6 weighted components of the enrichment score with values.
    """
    components = []

    # Livestock growth (20%)
    trend_avg = (
        row.get("livestock_large_trend", 0)
        + row.get("livestock_small_trend", 0)
        + row.get("milk_trend", 0)
    ) / 3
    growth_score = ((trend_avg + 1) / 2) * 20
    trend_word = "growing" if trend_avg > 0 else "declining" if trend_avg < 0 else "stable"
    components.append({
        "component": "Livestock Growth",
        "weight": "20%",
        "score": round(growth_score, 1),
        "max_score": 20,
        "text": f"Livestock herds are {trend_word} (avg trend: {trend_avg:+.3f})",
        "direction": "positive" if growth_score >= 10 else "negative",
    })

    # Farm scale (15%)
    # Approximate using raw values percentile (we don't have the full df here,
    # so we give a rough absolute rating)
    large = row.get("large_livestock_latest", 0)
    small = row.get("small_livestock_latest", 0)
    milk = row.get("milk_production_latest", 0)
    # Rough heuristic: log-scale normalization
    scale_score = min(15, (
        np.log1p(large) / np.log1p(2000) * 0.35
        + np.log1p(small) / np.log1p(15000) * 0.35
        + np.log1p(milk) / np.log1p(5000) * 0.30
    ) * 15)
    components.append({
        "component": "Farm Scale",
        "weight": "15%",
        "score": round(scale_score, 1),
        "max_score": 15,
        "text": f"{int(large)} large + {int(small)} small livestock, {milk:.0f}t milk",
        "direction": "positive" if scale_score >= 7.5 else "negative",
    })

    # Financial health (25%)
    revenue = row.get("annual_revenue", 0)
    credit = row.get("credit_score", 650)
    bankrupt = row.get("is_bankrupt", False)
    tax_debt = row.get("has_tax_debt", False)
    credit_norm = max(0, min(1, (credit - 300) / 550))
    no_bankrupt = 0.0 if bankrupt else 1.0
    no_tax = 0.0 if tax_debt else 1.0
    revenue_norm = min(1.0, np.log1p(revenue) / np.log1p(500_000_000))
    fin_score = (revenue_norm * 0.35 + credit_norm * 0.25 + no_bankrupt * 0.25 + no_tax * 0.15) * 25
    fin_issues = []
    if bankrupt:
        fin_issues.append("BANKRUPT")
    if tax_debt:
        fin_issues.append("tax debt")
    fin_text = f"Revenue ₸{revenue:,.0f}, credit {int(credit)}"
    if fin_issues:
        fin_text += f" — WARNING: {', '.join(fin_issues)}"
    components.append({
        "component": "Financial Health",
        "weight": "25%",
        "score": round(fin_score, 1),
        "max_score": 25,
        "text": fin_text,
        "direction": "positive" if fin_score >= 12.5 else "negative",
    })

    # Compliance (15%)
    ndvi = row.get("ndvi_score", 0.5)
    env = row.get("env_compliance", 0.5)
    comp_score = (ndvi * 0.5 + env * 0.5) * 15
    components.append({
        "component": "Compliance",
        "weight": "15%",
        "score": round(comp_score, 1),
        "max_score": 15,
        "text": f"NDVI: {ndvi:.3f}, environmental: {env:.3f}",
        "direction": "positive" if comp_score >= 7.5 else "negative",
    })

    # Legal standing (10%)
    lawsuits = row.get("active_lawsuits", 0)
    legal_score = max(0, (1 - lawsuits / 3)) * 10
    lawsuit_text = "clean legal record" if lawsuits == 0 else f"{int(lawsuits)} active lawsuit(s)"
    components.append({
        "component": "Legal Standing",
        "weight": "10%",
        "score": round(legal_score, 1),
        "max_score": 10,
        "text": lawsuit_text,
        "direction": "positive" if lawsuits == 0 else "negative",
    })

    # Track record (15%)
    approval = row.get("approval_rate", 0)
    track_score = approval * 0.6 * 15 + 0.4 * 15 * 0.5  # assume median history rank
    components.append({
        "component": "Track Record",
        "weight": "15%",
        "score": round(min(15, track_score), 1),
        "max_score": 15,
        "text": f"Approval rate: {approval:.0%}, {int(row.get('total_applications', 0))} applications",
        "direction": "positive" if approval >= 0.7 else "negative",
    })

    return components


def explain_all_applicants(
    model: CatBoostClassifier,
    scored_applicants: pd.DataFrame,
    transactions: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    top_n: int = 5,
) -> dict[str, dict]:
    """Generate full explanations (SHAP + enrichment breakdown) for all scored applicants.

    Returns:
        Dict mapping bin_iin -> {"shap": [...], "enrichment": [...]}
    """
    from src.model import ENRICHMENT_MERGE_COLS

    df = transactions.merge(
        scored_applicants[ENRICHMENT_MERGE_COLS], on="bin_iin", how="inner"
    )
    # Take the latest transaction per applicant
    df = df.sort_values("date_received").groupby("bin_iin").tail(1).reset_index(drop=True)

    X = df[feature_cols].copy()
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)
    for col in feature_cols:
        if col not in cat_features:
            X[col] = X[col].fillna(0)

    shap_vals = compute_shap_values(model, X)
    values = shap_vals.values
    if values.ndim == 3:
        values = values[:, :, 1]

    explanations = {}
    for i, bin_iin in enumerate(df["bin_iin"].values):
        shap_exps = explain_single_applicant(
            values[i], feature_cols, X.iloc[i], top_n=top_n,
        )
        # Get enrichment breakdown from scored_applicants
        applicant_row = scored_applicants[scored_applicants["bin_iin"] == bin_iin]
        if len(applicant_row) > 0:
            enrichment_exps = generate_enrichment_breakdown(applicant_row.iloc[0])
        else:
            enrichment_exps = []

        explanations[bin_iin] = {
            "shap": shap_exps,
            "enrichment": enrichment_exps,
        }

    return explanations


if __name__ == "__main__":
    from src.data_ingestion import load_and_process
    from src.enrichment import enrich_applicants
    from src.model import (
        prepare_training_data, train_model, compute_merit_scores,
        ALL_FEATURES, CATEGORICAL_FEATURES,
    )

    print("Loading data...")
    transactions, applicants = load_and_process()
    enriched = enrich_applicants(applicants)

    print("Training model...")
    train_df = prepare_training_data(transactions, enriched)
    model, _ = train_model(train_df, iterations=300)
    scores = compute_merit_scores(model, enriched, transactions)

    print("\nGenerating explanations for top 5...")
    explanations = explain_all_applicants(
        model, scores.head(50), transactions, ALL_FEATURES, CATEGORICAL_FEATURES
    )

    for bin_iin in list(explanations.keys())[:5]:
        score = scores.loc[scores["bin_iin"] == bin_iin, "merit_score"].values[0]
        exp = explanations[bin_iin]
        print(f"\n{'='*60}")
        print(f"Applicant {bin_iin} — Merit Score: {score}")
        print(f"\n  ML Factors (SHAP):")
        for s in exp["shap"]:
            print(f"    {'🟢' if s['direction']=='positive' else '🔴'} {s['text']}")
        print(f"\n  Enrichment Breakdown:")
        for e in exp["enrichment"]:
            icon = "🟢" if e["direction"] == "positive" else "🔴"
            print(f"    {icon} [{e['component']}] {e['score']}/{e['max_score']} — {e['text']}")
