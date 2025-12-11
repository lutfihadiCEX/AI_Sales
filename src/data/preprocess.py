import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from src.data.load_data import load_sales_data

def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the sales dataset:
    - Feature engineering (month, quarter, year)
    - Encode categorical columns
    
    """

    df = df.copy()

    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month
    df['order_quarter'] = df['order_date'].dt.quarter

    # Encoding stage
    categorical_cols = ['region', 'country', 'product_category',
                        'product_subcategory', 'customer_segment', 'marketing_channel']
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le # optional for inverse transform

    # Anomalies flagged
    threshold = df['total_sales'].mean() + 3 * df['total_sales'].std()
    df['is_anomaly'] = df['total_sales'] > threshold

    return df, le_dict

if __name__ == "__main__":
    raw_df = load_sales_data()
    processed_df, encoders = preprocess_sales_data(raw_df)
    print(processed_df.head())
    print(f"Columns after preprocessing: {processed_df.columns.tolist()}")