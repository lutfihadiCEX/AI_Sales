import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.ml.predict import run_prediction


st.set_page_config(
    page_title="AI Sales Analyst",
    layout="wide"
)

st.title("📊 AI Sales Analyst")
st.write(
    "Upload your sales CSV and get predictions, anomalies, and insights."
)

uploaded_file = st.file_uploader(
    "Upload Sales CSV",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully!")

        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        with st.spinner("Running AI analysis..."):
            results = run_prediction(df)

        st.subheader("Predictions & Anomalies")
        st.dataframe(results.head())

        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Rows",
            len(results)
        )
        col2.metric(
            "Anomalies Detected",
            int(results["is_anomaly"].sum())
        )
        col3.metric(
            "Avg RF Predicted Sales",
            round(results["rf_predicted_sales"].mean(), 2)
        )

        st.subheader("Sales vs Predicted (Random Forest)")
        st.line_chart(
            results[["total_sales", "rf_predicted_sales"]]
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
