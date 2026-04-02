"""
Phase 2: Data Enrichment — "360-degree view" of the farmer.

Generates deterministic synthetic data simulating real government/external
data sources for each applicant:

LIVESTOCK DATA (2021-2025):
- Large livestock count (cows, horses) per year
- Small livestock count (sheep, goats) per year
- Milk production (tons) per year
- Year-over-year growth trends computed from the series

FINANCIAL & LEGAL:
- Annual revenue (KZT)
- Bankruptcy status (binary)
- Tax debt status (binary)
- Credit score (300-850)

OPERATIONAL:
- Active lawsuits count
- Land use verification (NDVI satellite score)
- Environmental compliance score

Uses BIN/IIN as seed for reproducibility — same applicant always
gets the same enrichment data across runs.
"""

import hashlib

import numpy as np
import pandas as pd

YEARS = [2021, 2022, 2023, 2024, 2025]


def _bin_to_seed(bin_iin: str) -> int:
    """Convert BIN/IIN string to a deterministic integer seed."""
    return int(hashlib.md5(bin_iin.encode()).hexdigest()[:8], 16)


def _make_rng(bin_iin: str, salt: str = "") -> np.random.RandomState:
    """Create a seeded RNG for a given BIN and optional salt."""
    return np.random.RandomState(_bin_to_seed(bin_iin + salt))


# ---------------------------------------------------------------------------
# Livestock data generators
# ---------------------------------------------------------------------------

def generate_livestock_large(bin_iin: str) -> dict:
    """Generate large livestock counts (cows, horses) for 2021-2025.

    Simulates a farm that starts with some base herd and grows or shrinks
    over time depending on the farm's trajectory.
    """
    rng = _make_rng(bin_iin, "large_livestock")
    base = int(rng.lognormal(mean=3.5, sigma=1.2))  # 10-200 head typical
    base = max(5, min(base, 2000))

    counts = {}
    current = base
    for year in YEARS:
        # Year-over-year change: -20% to +25%, with slight positive bias
        yoy_change = rng.normal(0.03, 0.12)
        current = max(1, int(current * (1 + yoy_change)))
        counts[year] = current
    return counts


def generate_livestock_small(bin_iin: str) -> dict:
    """Generate small livestock counts (sheep, goats) for 2021-2025.

    Small livestock herds are typically 3-10x larger than large livestock.
    """
    rng = _make_rng(bin_iin, "small_livestock")
    base = int(rng.lognormal(mean=4.5, sigma=1.3))  # 50-1000 head typical
    base = max(10, min(base, 15000))

    counts = {}
    current = base
    for year in YEARS:
        yoy_change = rng.normal(0.04, 0.15)
        current = max(1, int(current * (1 + yoy_change)))
        counts[year] = current
    return counts


def generate_milk_production(bin_iin: str, large_livestock: dict) -> dict:
    """Generate milk production (tons) for 2021-2025.

    Correlated with large livestock count — roughly 2-5 tons per cow/year,
    but with noise and efficiency variation.
    """
    rng = _make_rng(bin_iin, "milk")
    # Yield per head varies by farm quality (tons/head/year)
    yield_per_head = rng.uniform(1.5, 5.0)

    production = {}
    for year in YEARS:
        heads = large_livestock.get(year, 50)
        # Not all large livestock are dairy — ~30-70% are milking
        dairy_fraction = rng.uniform(0.25, 0.70)
        base_production = heads * dairy_fraction * yield_per_head
        # Add noise +-15%
        noise = rng.normal(1.0, 0.15)
        production[year] = round(max(0.1, base_production * noise), 1)
    return production


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------

def compute_trend(yearly_data: dict) -> float:
    """Compute a normalized growth trend from yearly data.

    Returns a value roughly in [-1, 1]:
    - Positive = growing operation
    - Negative = shrinking operation
    - 0 = stable

    Uses linear regression slope normalized by mean.
    """
    values = [yearly_data[y] for y in YEARS]
    if len(values) < 2 or np.mean(values) == 0:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    normalized = slope / np.mean(values)
    return round(float(np.clip(normalized, -1.0, 1.0)), 3)


# ---------------------------------------------------------------------------
# Financial & legal generators
# ---------------------------------------------------------------------------

def generate_revenue(bin_iin: str, large_livestock: dict, small_livestock: dict) -> float:
    """Generate annual revenue (KZT) correlated with herd size.

    Revenue comes from: livestock sales, dairy, subsidies, crop sales.
    Rough estimate: 200-500k KZT per large head, 50-150k per small head.
    """
    rng = _make_rng(bin_iin, "revenue")
    large_latest = large_livestock.get(2025, 50)
    small_latest = small_livestock.get(2025, 100)

    per_large = rng.uniform(200_000, 500_000)
    per_small = rng.uniform(50_000, 150_000)
    base_revenue = large_latest * per_large + small_latest * per_small

    # Add diversification bonus (crop sales etc) — 0-40% extra
    diversification = rng.uniform(1.0, 1.4)
    # Annual variation +-20%
    variation = rng.normal(1.0, 0.2)
    revenue = base_revenue * diversification * max(0.3, variation)
    return round(revenue, 0)


def generate_bankruptcy(bin_iin: str) -> bool:
    """Simulate bankruptcy status. ~5% of applicants are bankrupt."""
    rng = _make_rng(bin_iin, "bankruptcy")
    return bool(rng.random() < 0.05)


def generate_tax_debt(bin_iin: str) -> bool:
    """Simulate tax debt status. ~15% of applicants have tax debt."""
    rng = _make_rng(bin_iin, "tax")
    return bool(rng.random() < 0.15)


def generate_judicial_history(bin_iin: str) -> int:
    """Simulate number of active lawsuits. Most have 0, max ~3."""
    rng = _make_rng(bin_iin, "judicial")
    return int(rng.choice([0, 0, 0, 0, 1, 1, 1, 2, 2, 3], size=1)[0])


def generate_ndvi_score(bin_iin: str) -> float:
    """Simulate land-use / NDVI verification score (0.0 to 1.0)."""
    rng = _make_rng(bin_iin, "ndvi")
    return round(float(rng.beta(5, 2)), 3)


def generate_credit_score(bin_iin: str) -> int:
    """Simulate credit score (300-850 range)."""
    rng = _make_rng(bin_iin, "credit")
    return int(np.clip(rng.normal(650, 80), 300, 850))


def generate_env_compliance(bin_iin: str) -> float:
    """Simulate environmental compliance score (0.0 to 1.0)."""
    rng = _make_rng(bin_iin, "env")
    return round(float(rng.beta(7, 2)), 3)


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------

def enrich_single_applicant(bin_iin: str) -> dict:
    """Generate all enrichment data for a single applicant.

    Returns a flat dict ready to be merged into a DataFrame row.
    """
    large = generate_livestock_large(bin_iin)
    small = generate_livestock_small(bin_iin)
    milk = generate_milk_production(bin_iin, large)

    result = {"bin_iin": bin_iin}

    # Yearly livestock counts
    for year in YEARS:
        result[f"large_livestock_{year}"] = large[year]
        result[f"small_livestock_{year}"] = small[year]
        result[f"milk_production_{year}"] = milk[year]

    # Latest values (for model features)
    result["large_livestock_latest"] = large[2025]
    result["small_livestock_latest"] = small[2025]
    result["milk_production_latest"] = milk[2025]

    # Trends
    result["livestock_large_trend"] = compute_trend(large)
    result["livestock_small_trend"] = compute_trend(small)
    result["milk_trend"] = compute_trend(milk)

    # Financial & legal
    result["annual_revenue"] = generate_revenue(bin_iin, large, small)
    result["is_bankrupt"] = generate_bankruptcy(bin_iin)
    result["has_tax_debt"] = generate_tax_debt(bin_iin)
    result["credit_score"] = generate_credit_score(bin_iin)

    # Operational
    result["active_lawsuits"] = generate_judicial_history(bin_iin)
    result["ndvi_score"] = generate_ndvi_score(bin_iin)
    result["env_compliance"] = generate_env_compliance(bin_iin)

    return result


def enrich_applicants(applicants: pd.DataFrame) -> pd.DataFrame:
    """Add all enrichment columns to the applicant dataframe.

    Args:
        applicants: DataFrame with 'bin_iin' column from data_ingestion.

    Returns:
        Enriched DataFrame with livestock, financial, legal, and operational data.
    """
    enrichment_rows = [
        enrich_single_applicant(bin_iin)
        for bin_iin in applicants["bin_iin"]
    ]
    enrichment_df = pd.DataFrame(enrichment_rows)
    df = applicants.merge(enrichment_df, on="bin_iin", how="left")
    return df


if __name__ == "__main__":
    from src.data_ingestion import load_and_process

    print("Loading data...")
    _, applicants = load_and_process()
    enriched = enrich_applicants(applicants)

    print(f"Enriched {len(enriched)} applicants")
    print(f"\n--- Livestock Stats ---")
    print(f"Large livestock (2025 avg): {enriched['large_livestock_latest'].mean():.0f}")
    print(f"Small livestock (2025 avg): {enriched['small_livestock_latest'].mean():.0f}")
    print(f"Milk production (2025 avg): {enriched['milk_production_latest'].mean():.1f} tons")
    print(f"Large livestock trend (avg): {enriched['livestock_large_trend'].mean():.3f}")
    print(f"Small livestock trend (avg): {enriched['livestock_small_trend'].mean():.3f}")
    print(f"Milk trend (avg): {enriched['milk_trend'].mean():.3f}")

    print(f"\n--- Financial & Legal ---")
    print(f"Revenue (avg): {enriched['annual_revenue'].mean():,.0f} KZT")
    print(f"Bankrupt: {enriched['is_bankrupt'].mean():.1%}")
    print(f"Tax debt: {enriched['has_tax_debt'].mean():.1%}")
    print(f"Credit score (avg): {enriched['credit_score'].mean():.0f}")

    print(f"\n--- Operational ---")
    print(f"Lawsuits (avg): {enriched['active_lawsuits'].mean():.2f}")
    print(f"NDVI (avg): {enriched['ndvi_score'].mean():.3f}")
    print(f"Env compliance (avg): {enriched['env_compliance'].mean():.3f}")

    print(f"\nSample (first 3):")
    sample_cols = ["bin_iin", "large_livestock_latest", "small_livestock_latest",
                   "milk_production_latest", "livestock_large_trend", "annual_revenue",
                   "is_bankrupt", "has_tax_debt"]
    print(enriched[sample_cols].head(3).to_string())
