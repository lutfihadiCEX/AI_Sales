from pathlib import Path
from typing import Iterable, Dict, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.load_data import load_sales_data


CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "region",
    "country",
    "product_category",
    "product_subcategory",
    "customer_segment",
    "marketing_channel",
)


def add_date_features(df: pd.DataFrame, date_column: str) -> None:
    """Add year, month, and quarter features."""
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    df["order_year"] = df[date_column].dt.year
    df["order_month"] = df[date_column].dt.month
    df["order_quarter"] = df[date_column].dt.quarter


def encode_categorical_features(df: pd.DataFrame, columns: Iterable[str],) -> Dict[str, LabelEncoder]:
    """Encode categorical columns and return encoders."""
    encoders: Dict[str, LabelEncoder] = {}

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Missing categorical column: {col}")

        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    return encoders


def flag_anomalies(df: pd.DataFrame, target_column: str, sigma: float = 3.0,) -> None:
    """Flag anomalies using mean ± sigma * std."""
    mean = df[target_column].mean()
    std = df[target_column].std()

    threshold = mean + sigma * std
    df["is_anomaly"] = df[target_column] > threshold


def preprocess_sales_data(df: pd.DataFrame,) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Apply feature engineering, encoding, and anomaly detection.
    """
    processed_df = df.copy()

    add_date_features(processed_df, "order_date")

    encoders = encode_categorical_features(
        processed_df,
        CATEGORICAL_COLUMNS,
    )

    flag_anomalies(processed_df, "total_sales")

    return processed_df, encoders


if __name__ == "__main__":
    raw_df = load_sales_data()
    processed_df, encoders = preprocess_sales_data(raw_df)

    print(processed_df.head())
    print(f"Columns after preprocessing: {processed_df.columns.tolist()}")
