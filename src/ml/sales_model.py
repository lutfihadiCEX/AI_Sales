from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.data.load_data import load_sales_data
from src.data.preprocess import preprocess_sales_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "total_sales"
DROP_COLUMNS = (
    "order_id",
    "order_date",
    "product_name",
    "is_anomaly",
    TARGET_COLUMN,
)


def split_features_target(df: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    X = df.drop(columns=list(DROP_COLUMNS))
    y = df[TARGET_COLUMN]
    return X, y


def get_models() -> Dict[str, object]:
    """Initialize models."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        ),
    }


def evaluate_model(y_true: pd.Series, y_pred: pd.Series,) -> Dict[str, float]:
    """Evaluate regression performance."""
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_and_evaluate(models: Dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Train models and return evaluation metrics."""
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = evaluate_model(y_test, predictions)
        results[name] = metrics

        print(f"\n{name.upper()}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R2:   {metrics['r2']:.4f}")

    return results


def save_artifacts(models: Dict[str, object], encoders: Dict,) -> None:
    """Persist trained models and encoders."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")


def train_models() -> Dict[str, object]:
    """Main training pipeline."""
    raw_df = load_sales_data()
    processed_df, encoders = preprocess_sales_data(raw_df)

    X, y = split_features_target(processed_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()

    train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_artifacts(models, encoders)

    return models


if __name__ == "__main__":
    trained_models = train_models()
