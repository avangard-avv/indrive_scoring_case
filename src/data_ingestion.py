"""
Phase 1: Data Ingestion & Processing Pipeline.

Downloads the Kazakhstan Agriculture Subsidy Registry from Google Sheets,
cleans Cyrillic text, currency formatting, and datetime fields.
"""

import io

import pandas as pd
import requests

DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1ejmCW0Lkdg561TcwWNvp-q38whtPFlVS/export?format=csv"
)

# Canonical column names (mapped from the Kazakh/Russian headers)
COLUMN_MAP = {
    "№ п/п": "row_number",
    "Дата поступления": "date_received",
    "Область": "region",
    "Акимат": "akimat",
    "Номер заявки": "application_id",
    "Направление водства": "direction",
    "Наименование субсидирования": "subsidy_name",
    "Статус заявки": "status",
    "Норматив": "norm_amount",
    "Причитающая сумма": "requested_amount",
    "Район хозяйства": "district",
}

STATUS_MAP = {
    "Исполнена": "Executed",
    "Отклонена": "Rejected",
    "Отозвано": "Withdrawn",
    "Одобрена": "Approved",
    "Сформировано поручение": "Order Formed",
    "Получена": "Received",
}


def download_csv(url: str = DATA_URL) -> pd.DataFrame:
    """Download the raw CSV from the remote URL."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    # Read with no header first — the real header is at row index 4
    df = pd.read_csv(
        io.BytesIO(resp.content), header=None, encoding="utf-8"
    )
    return df


def parse_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the actual header and data rows from the raw download."""
    # Row 4 (0-indexed) contains the column names
    header_row = df.iloc[4]
    data = df.iloc[5:].copy()
    data.columns = header_row.values

    # Drop the two empty columns (indices 2 and 3 in the original)
    cols_to_keep = [c for c in data.columns if pd.notna(c)]
    data = data[cols_to_keep].copy()
    data = data.reset_index(drop=True)
    return data


def clean_currency(series: pd.Series) -> pd.Series:
    """Convert Kazakh-formatted currency strings to float.

    Handles non-breaking spaces (\xa0) as thousand separators
    and commas as decimal separators.
    """
    cleaned = (
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and rename columns to English."""
    # Rename columns
    df = df.rename(columns=COLUMN_MAP)

    # Clean currency fields
    for col in ["requested_amount", "norm_amount"]:
        if col in df.columns:
            df[col] = clean_currency(df[col])

    # Parse datetime
    df["date_received"] = pd.to_datetime(
        df["date_received"], format="%d.%m.%Y %H:%M:%S", errors="coerce"
    )

    # Map statuses to English
    df["status_en"] = df["status"].map(STATUS_MAP).fillna("Unknown")

    # Extract BIN/IIN from application_id (first 12 digits)
    df["bin_iin"] = df["application_id"].astype(str).str[:12]

    # Drop the raw row_number column
    df = df.drop(columns=["row_number"], errors="ignore")

    # Remove rows with no application_id (metadata/footer rows)
    df = df.dropna(subset=["application_id"]).reset_index(drop=True)

    return df


def extract_applicant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-applicant (BIN/IIN) features from transaction history."""
    agg = df.groupby("bin_iin").agg(
        total_applications=("application_id", "count"),
        total_requested=("requested_amount", "sum"),
        avg_requested=("requested_amount", "mean"),
        executed_count=("status_en", lambda x: (x == "Executed").sum()),
        rejected_count=("status_en", lambda x: (x == "Rejected").sum()),
        first_application=("date_received", "min"),
        last_application=("date_received", "max"),
        primary_region=("region", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
        primary_direction=("direction", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
        primary_district=("district", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
        unique_directions=("direction", "nunique"),
    ).reset_index()

    # Derived features
    agg["approval_rate"] = agg["executed_count"] / agg["total_applications"]
    agg["rejection_rate"] = agg["rejected_count"] / agg["total_applications"]
    agg["history_days"] = (
            agg["last_application"] - agg["first_application"]
    ).dt.days.fillna(0)

    return agg


def load_and_process() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: download, parse, clean, and extract features.

    Returns:
        (cleaned_transactions, applicant_features)
    """
    raw = download_csv()
    parsed = parse_raw(raw)
    cleaned = clean_dataframe(parsed)
    applicants = extract_applicant_features(cleaned)
    return cleaned, applicants


if __name__ == "__main__":
    print("Downloading and processing data...")
    transactions, applicants = load_and_process()
    print(f"Transactions: {len(transactions)} rows")
    print(f"Unique applicants: {len(applicants)}")
    print(f"\nStatus distribution:\n{transactions['status_en'].value_counts()}")
    print(f"\nSample applicant features:\n{applicants.head()}")
