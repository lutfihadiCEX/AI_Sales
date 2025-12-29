from pathlib import Path
import pandas as pd
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "sales_data.csv"

NUMERIC_COLUMNS: tuple[str, ...] = (
    "unit_price",
    "quantity",
    "discount",
    "total_sales",
    "inventory_level",
    "marketing_spend",
    "previous_month_sales",
)


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {path}")

    return pd.read_csv(path)


def parse_dates(df: pd.DataFrame, column: str) -> None:
    """Parse date column in place."""
    df[column] = pd.to_datetime(df[column], errors="coerce")


def parse_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Convert columns to numeric in place."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def drop_invalid_rows(df: pd.DataFrame, required_columns: Iterable[str],) -> pd.DataFrame:
    """Drop rows with missing required values."""
    return df.dropna(subset=list(required_columns)).reset_index(drop=True)


def load_sales_data(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and clean sales data.
    """
    path = csv_path or DEFAULT_CSV_PATH

    df = load_csv(path)

    parse_dates(df, "order_date")
    parse_numeric_columns(df, NUMERIC_COLUMNS)

    required_cols = ("order_id", "order_date", *NUMERIC_COLUMNS)
    df = drop_invalid_rows(df, required_cols)

    return df


if __name__ == "__main__":
    df = load_sales_data()
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    print(df.head())
