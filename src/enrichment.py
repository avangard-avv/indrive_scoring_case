"""
Phase 2: Data Enrichment — Mock "360-degree view" of the farmer.

Generates deterministic synthetic data to simulate external data sources:
- Tax Debt Status (binary)
- Judicial History (number of active lawsuits)
- Land Use Verification (simulated NDVI score)
- Credit Score (synthetic)
- Environmental Compliance (synthetic)

Uses BIN/IIN as seed for reproducibility — same applicant always
gets the same enrichment data.
"""

import hashlib

import numpy as np
import pandas as pd


def _bin_to_seed(bin_iin: str) -> int:
    """Convert BIN/IIN string to a deterministic integer seed."""
    return int(hashlib.md5(bin_iin.encode()).hexdigest()[:8], 16)


def generate_tax_debt(bin_iin: str) -> bool:
    """Simulate tax debt status. ~15% of applicants have tax debt."""
    rng = np.random.RandomState(_bin_to_seed(bin_iin))
    return bool(rng.random() < 0.15)


def generate_judicial_history(bin_iin: str) -> int:
    """Simulate number of active lawsuits. Most have 0, max ~3."""
    rng = np.random.RandomState(_bin_to_seed(bin_iin))
    # 80% zero, 15% one, 4% two, 1% three
    return int(rng.choice([0, 0, 0, 0, 1, 1, 1, 2, 2, 3], size=1)[0])


def generate_ndvi_score(bin_iin: str) -> float:
    """Simulate a land-use / NDVI verification score (0.0 to 1.0).

    Higher means better land utilization (greener fields, active farming).
    """
    rng = np.random.RandomState(_bin_to_seed(bin_iin))
    # Beta distribution skewed toward higher values (most farms are active)
    return round(float(rng.beta(5, 2)), 3)


def generate_credit_score(bin_iin: str) -> int:
    """Simulate a credit score (300-850 range, similar to FICO)."""
    rng = np.random.RandomState(_bin_to_seed(bin_iin))
    return int(np.clip(rng.normal(650, 80), 300, 850))


def generate_environmental_compliance(bin_iin: str) -> float:
    """Simulate environmental compliance score (0.0 to 1.0)."""
    rng = np.random.RandomState(_bin_to_seed(bin_iin))
    return round(float(rng.beta(7, 2)), 3)


def enrich_applicants(applicants: pd.DataFrame) -> pd.DataFrame:
    """Add all synthetic enrichment columns to the applicant dataframe.

    Args:
        applicants: DataFrame with 'bin_iin' column from data_ingestion.

    Returns:
        Enriched DataFrame with additional columns.
    """
    df = applicants.copy()

    df["has_tax_debt"] = df["bin_iin"].apply(generate_tax_debt)
    df["active_lawsuits"] = df["bin_iin"].apply(generate_judicial_history)
    df["ndvi_score"] = df["bin_iin"].apply(generate_ndvi_score)
    df["credit_score"] = df["bin_iin"].apply(generate_credit_score)
    df["env_compliance"] = df["bin_iin"].apply(generate_environmental_compliance)

    return df


if __name__ == "__main__":
    from src.data_ingestion import load_and_process

    print("Loading data...")
    _, applicants = load_and_process()
    enriched = enrich_applicants(applicants)

    print(f"Enriched {len(enriched)} applicants")
    print(f"\nTax debt prevalence: {enriched['has_tax_debt'].mean():.1%}")
    print(f"Avg lawsuits: {enriched['active_lawsuits'].mean():.2f}")
    print(f"Avg NDVI score: {enriched['ndvi_score'].mean():.3f}")
    print(f"Avg credit score: {enriched['credit_score'].mean():.0f}")
    print(f"\nSample:\n{enriched[['bin_iin', 'has_tax_debt', 'active_lawsuits', 'ndvi_score', 'credit_score', 'env_compliance']].head(10)}")
