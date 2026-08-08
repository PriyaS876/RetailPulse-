import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_models():
    churn_model = joblib.load("Models/churn_model.pkl")
    kmeans_model = joblib.load("Models/kmeans_model.pkl")
    prophet_model = joblib.load("Models/prophet_model.pkl")
    rfm_scaler = joblib.load("Models/rfm_scaler.pkl")

    return churn_model, kmeans_model, prophet_model, rfm_scaler


churn_model, kmeans_model, prophet_model, rfm_scaler = load_models()

# Sidebar
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "❤️ Churn Prediction",
        "👥 Customer Segmentation",
        "📈 Sales Forecasting"
    ]
)
# ---------------- HOME ----------------
if page == "🏠 Home":

    st.title("📊 RetailPulse")
    st.subheader(
        "AI-Powered Customer Analytics & Demand Forecasting Platform"
    )

    st.write(
        "RetailPulse combines customer analytics, machine learning, "
        "customer segmentation and sales forecasting to support "
        "data-driven business decisions."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🤖 ML Models", "3")

    with col2:
        st.metric("👥 Customer Analytics", "RFM + Segmentation")

    with col3:
        st.metric("📈 Forecasting", "Prophet")

    st.success("All trained models loaded successfully! ✅")


# ---------------- CHURN ----------------


elif page == "❤️ Churn Prediction":

    st.title("❤️ Customer Churn Prediction")

    st.write(
        "Enter customer information to predict whether the customer "
        "is likely to churn."
    )

    st.divider()

    frequency = st.number_input(
        "Purchase Frequency",
        min_value=0.0,
        value=5.0
    )

    monetary = st.number_input(
        "Total Monetary Value",
        min_value=0.0,
        value=500.0
    )

    if st.button("🔮 Predict Churn"):

        input_data = np.array([[frequency, monetary]])

        prediction = churn_model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ Customer is predicted to CHURN.")
        else:
            st.success("✅ Customer is predicted to STAY.")


# ---------------- CUSTOMER SEGMENTATION ----------------

# ---------------- CUSTOMER SEGMENTATION ----------------
elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.write(
        "Enter customer RFM values to identify the customer segment "
        "using the trained K-Means model."
    )

    st.divider()

    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0.0,
        value=30.0
    )

    frequency = st.number_input(
        "Frequency (number of purchases)",
        min_value=0.0,
        value=5.0
    )

    monetary = st.number_input(
        "Monetary (total spending)",
        min_value=0.0,
        value=500.0
    )

    if st.button("🔍 Identify Customer Segment"):

        input_data = np.array([
            [recency, frequency, monetary]
        ])

        # Apply the same scaler used during training
        scaled_input = rfm_scaler.transform(input_data)

        # Predict customer cluster
        cluster = kmeans_model.predict(scaled_input)[0]

        st.success(
            f"Customer belongs to Segment / Cluster: {cluster}"
        )
# ---------------- SALES FORECASTING ----------------

elif page == "📈 Sales Forecasting":

    st.title("📈 Sales Forecasting")

    st.write(
        "Forecast future monthly sales using the trained Prophet model."
    )

    st.divider()

    months = st.slider(
        "Forecast Horizon (months)",
        min_value=1,
        max_value=12,
        value=6
    )

    if st.button("📈 Generate Forecast"):

        # Create future monthly dates
        future = prophet_model.make_future_dataframe(
            periods=months,
            freq="ME"
        )

        # Generate forecast
        forecast = prophet_model.predict(future)

        # Keep future forecast only
        forecast_future = forecast.tail(months)

        st.subheader("Future Sales Forecast")

        st.dataframe(
            forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        )

        st.line_chart(
            forecast_future.set_index("ds")["yhat"]
        )