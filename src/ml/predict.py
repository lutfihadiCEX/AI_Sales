from pathlib import Path
import joblib
import pandas as pd
from src.data.preprocess import preprocess_sales_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

def load_models():
    """Load trained models/encoders"""
    lr_model = joblib.load(MODELS_DIR / "linear_regression_model.pkl")
    rf_model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model.pkl" )
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")

    return lr_model, rf_model, xgb_model, encoders

def run_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prediction on new datasets. Returns in dataframe form

    """
    lr_model, rf_model, xgb_model, encoders = load_models()

    df_processed, _ = preprocess_sales_data(df, encoders=encoders)

    drop_cols = [
        "order_id",
        "order_date",
        "total_sales",
        "is_anomaly",
        "product_name",
    ]

    X = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns])

    df_processed["lr_predicted_sales"] = lr_model.predict(X)
    df_processed["rf_predicted_sales"] = rf_model.predict(X)
    df_processed["xgb_predicted_sales"] = xgb_model.predict(X)

    return df_processed

if __name__ == "__main__":
    from src.data.load_data import load_sales_data

    raw_df = load_sales_data()
    predictions = run_prediction(raw_df)

    print(predictions[
        ["order_id", "total_sales", "lr_predicted_sales", "rf_predicted_sales", "xgb_predicted_sales", "is_anomaly"]
    ].head())