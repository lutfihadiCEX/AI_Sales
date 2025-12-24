from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from src.data.load_data import load_sales_data
from src.data.preprocess import preprocess_sales_data


def train_models():

    df_raw = load_sales_data()
    df, encoders = preprocess_sales_data(df_raw)

    # Features and target
    X = df.drop(columns=['order_id', 'order_date', 'total_sales', 'is_anomaly', 'product_name'])
    y = df['total_sales']

    # TT-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simple Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # XGB 
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             objective="reg:squarederror", random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # Simple eval
    print("Linear Regression RMSE:", root_mean_squared_error(y_test, y_pred_lr))
    print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

    print("Random Forest RMSE:", root_mean_squared_error(y_test, y_pred_rf))
    print("Random Forest R2:", r2_score(y_test, y_pred_rf))

    print("\nXGBoost RMSE:", root_mean_squared_error(y_test, y_pred_xgb))
    print("XGBoost R2:", r2_score(y_test, y_pred_xgb))

    models_path = Path(__file__).resolve().parents[2] / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    # Save for use
    joblib.dump(lr_model, models_path / "linear_regression_model.pkl")
    joblib.dump(rf_model, models_path / "random_forest_model.pkl")
    joblib.dump(xgb_model, models_path / "xgboost_model.pkl")
    joblib.dump(encoders, models_path / "encoders.pkl")


    return lr_model, rf_model, xgb_model, encoders


if __name__ == "__main__":
    lr, rf, xgb, encoders = train_models()