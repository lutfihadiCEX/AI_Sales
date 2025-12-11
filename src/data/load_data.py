import pandas as pd
from pathlib import Path

def load_sales_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load the sales CSV file into a pandas DataFrame.
    Performs basic validation and type conversion.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned sales dataset.
    """

    if csv_path is None:
        project_root = Path(__file__).resolve().parents[2]  
        csv_path = project_root / "data" / "raw" / "sales_data.csv"

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    numeric_cols = ['unit_price', 'quantity', 'discount', 'total_sales',
                    'inventory_level', 'marketing_spend', 'previous_month_sales']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['order_id', 'order_date'] + numeric_cols)

    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {csv_path}")
    
    return df

if __name__ == "__main__":
    df = load_sales_data()
    print(df.head())