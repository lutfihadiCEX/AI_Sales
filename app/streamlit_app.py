import sys
from pathlib import Path
import matplotlib.pyplot as plt
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.ml.predict import run_prediction, load_models

def get_executive_summary(df):
    llm = ChatOllama(model="llama3.2", temperature=0.2)   # Low temp for factual analysis
    
    summary_stats = {
        "total_sales": df['total_sales'].sum(),
        "avg_discount": df['discount'].mean(),
        "anomaly_count": int(df['is_anomaly'].sum()),
        "top_region": df.groupby('region')['total_sales'].sum().idxmax()
    }
    
    prompt = f"""
    You are an AI Business Analyst. Based on these sales results:
    - Total Sales: ${summary_stats['total_sales']:,.2f}
    - Average Discount: {summary_stats['avg_discount']:.2%}
    - Anomalies: {summary_stats['anomaly_count']}
    - Top Performing Region (Encoded): {summary_stats['top_region']}
    
    Provide 3 high-level business insights and 1 warning about the anomalies.
    """
    
    return llm.invoke(prompt).content


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

        if st.checkbox("Show Sales Drivers (Feature Importance)"):
            # XGB feature importance vizualizer
            _, _, xgb_model, _ = load_models()
    
            feature_names = results.drop(columns=[c for c in ["order_id", "order_date", "total_sales", "is_anomaly", "product_name", "lr_predicted_sales", "rf_predicted_sales", "xgb_predicted_sales"] if c in results.columns]).columns
    
            fig, ax = plt.subplots()
            importance = pd.Series(xgb_model.feature_importances_, index=feature_names)
            importance.nlargest(10).plot(kind='barh', ax=ax)
            st.pyplot(fig)

        if st.button("🤖 Generate Executive Report"):
            with st.spinner("Llama 3.2 is analyzing your data..."):
                report = get_executive_summary(results)
                st.info(report)

    except Exception as e:
        st.error(f"Error processing file: {e}")
