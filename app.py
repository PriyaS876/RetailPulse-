import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# BASE PATH
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def find_file(folder, filename):
    path = project_path(folder, filename)

    if os.path.exists(path):
        return path

    return None


# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():

    churn_path = project_path(
        "Models",
        "churn_model.pkl"
    )

    kmeans_path = project_path(
        "Models",
        "kmeans_model.pkl"
    )

    prophet_path = project_path(
        "Models",
        "prophet_model.pkl"
    )

    scaler_path = project_path(
        "Models",
        "rfm_scaler.pkl"
    )

    churn_model = joblib.load(churn_path)
    kmeans_model = joblib.load(kmeans_path)
    prophet_model = joblib.load(prophet_path)
    rfm_scaler = joblib.load(scaler_path)

    return (
        churn_model,
        kmeans_model,
        prophet_model,
        rfm_scaler
    )


try:

    (
        churn_model,
        kmeans_model,
        prophet_model,
        rfm_scaler
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        f"Model loading error: {e}"
    )

    churn_model = None
    kmeans_model = None
    prophet_model = None
    rfm_scaler = None


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    data = {}

    files = {

        "retail":
            "Retail_Cleaned.csv",

        "rfm":
            "Customer_RFM.csv",

        "segments":
            "Customer_Segments.csv",

        "sales_forecast":
            "Sales_Forecast.csv",

        "lstm_forecast":
            "LSTM_Future_Forecast.csv",

        "churn":
            "Customer_Churn_Predictions.csv",

        # Optional files
        "prophet_daily":
            "Prophet_Daily_Future_Forecast.csv",

        "hybrid":
            "Hybrid_Future_Forecast.csv",

        "inventory":
            "Inventory_Recommendations.csv"
    }

    for key, filename in files.items():

        path = find_file(
            "Notebooks",
            filename
        )

        if path is not None:

            try:

                data[key] = pd.read_csv(path)

            except Exception as e:

                st.warning(
                    f"Could not load {filename}: {e}"
                )

                data[key] = None

        else:

            data[key] = None

    return data


data = load_data()


retail_df = data["retail"]
rfm_df = data["rfm"]
segments_df = data["segments"]
sales_forecast_df = data["sales_forecast"]
lstm_forecast_df = data["lstm_forecast"]
churn_df = data["churn"]
prophet_daily_df = data["prophet_daily"]
hybrid_df = data["hybrid"]
inventory_df = data["inventory"]


# ==================================================
# CREATE HYBRID FORECAST IF CSV DOES NOT EXIST
# ==================================================

if hybrid_df is None:

    if (
        prophet_daily_df is not None
        and lstm_forecast_df is not None
    ):

        try:

            prophet_daily_df["ds"] = pd.to_datetime(
                prophet_daily_df["ds"]
            )

            lstm_forecast_df["ds"] = pd.to_datetime(
                lstm_forecast_df["ds"]
            )

            hybrid_df = pd.merge(
                prophet_daily_df[
                    [
                        "ds",
                        "yhat",
                        "yhat_lower",
                        "yhat_upper"
                    ]
                ],
                lstm_forecast_df[
                    [
                        "ds",
                        "PredictedSales"
                    ]
                ],
                on="ds",
                how="inner"
            )

            hybrid_df["HybridPrediction"] = (
                hybrid_df["yhat"]
                + hybrid_df["PredictedSales"]
            ) / 2

        except Exception:

            hybrid_df = None


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📊 RetailPulse")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Sales Analytics",
        "❤️ Churn Prediction",
        "👥 Customer Segmentation",
        "📈 Sales Forecasting",
        "📦 Inventory Optimization",
        "💡 Business Insights"
    ]
)


# ==================================================
# HOME
# ==================================================

if page == "🏠 Home":

    st.title("📊 RetailPulse")

    st.subheader(
        "AI-Powered Customer Analytics & Demand Forecasting Platform"
    )

    st.write(
        "RetailPulse combines customer analytics, "
        "machine learning, customer segmentation, "
        "churn prediction, demand forecasting and "
        "inventory optimization to support "
        "data-driven business decisions."
    )

    st.divider()

    # ----------------------------------------------
    # KPI CALCULATION
    # ----------------------------------------------

    if retail_df is not None:

        df = retail_df.copy()

        # Create TotalAmount if missing
        if "TotalAmount" not in df.columns:

            if (
                "Quantity" in df.columns
                and "Price" in df.columns
            ):

                df["TotalAmount"] = (
                    df["Quantity"]
                    * df["Price"]
                )

        if "TotalAmount" in df.columns:

            total_sales = (
                df["TotalAmount"]
                .sum()
            )

        else:

            total_sales = 0

        if "Customer ID" in df.columns:

            total_customers = (
                df["Customer ID"]
                .nunique()
            )

        else:

            total_customers = 0

        if "Invoice" in df.columns:

            total_orders = (
                df["Invoice"]
                .nunique()
            )

        else:

            total_orders = len(df)

        if total_orders > 0:

            aov = (
                total_sales
                / total_orders
            )

        else:

            aov = 0

    else:

        total_sales = 0
        total_customers = 0
        total_orders = 0
        aov = 0

    # ----------------------------------------------
    # KPI CARDS
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💰 Total Sales",
            f"₹{total_sales:,.2f}"
        )

    with col2:

        st.metric(
            "👥 Customers",
            f"{total_customers:,}"
        )

    with col3:

        st.metric(
            "🧾 Orders",
            f"{total_orders:,}"
        )

    with col4:

        st.metric(
            "🛒 Average Order Value",
            f"₹{aov:,.2f}"
        )

    st.divider()

    # ----------------------------------------------
    # PLATFORM FEATURES
    # ----------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "🤖 Machine Learning\n\n"
            "Churn Prediction"
        )

    with col2:

        st.info(
            "👥 Customer Analytics\n\n"
            "RFM + Segmentation"
        )

    with col3:

        st.info(
            "📈 Forecasting\n\n"
            "LSTM + Prophet + Hybrid"
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "📦 Inventory Optimization\n\n"
            "Reorder Recommendations"
        )

    with col2:

        st.info(
            "📊 Sales Analytics\n\n"
            "Trends + Country Analysis"
        )

    with col3:

        st.info(
            "💡 Business Insights\n\n"
            "Actionable Recommendations"
        )

    if models_loaded:

        st.success(
            "All available trained models loaded successfully! ✅"
        )

    else:

        st.warning(
            "Some trained models could not be loaded."
        )


# ==================================================
# SALES ANALYTICS
# ==================================================

elif page == "📊 Sales Analytics":

    st.title("📊 Sales Analytics")

    st.write(
        "Explore historical sales performance, "
        "monthly trends and country-level sales."
    )

    st.divider()

    if retail_df is None:

        st.error(
            "Retail_Cleaned.csv not found."
        )

    else:

        df = retail_df.copy()

        # ------------------------------------------
        # TOTAL AMOUNT
        # ------------------------------------------

        if "TotalAmount" not in df.columns:

            if (
                "Quantity" in df.columns
                and "Price" in df.columns
            ):

                df["TotalAmount"] = (
                    df["Quantity"]
                    * df["Price"]
                )

            else:

                st.error(
                    "Quantity or Price column is missing."
                )

                st.stop()

        # ------------------------------------------
        # DATE
        # ------------------------------------------

        if "InvoiceDate" in df.columns:

            df["InvoiceDate"] = pd.to_datetime(
                df["InvoiceDate"],
                errors="coerce"
            )

        # ------------------------------------------
        # SALES OVERVIEW
        # ------------------------------------------

        st.subheader(
            "💰 Sales Overview"
        )

        total_sales = (
            df["TotalAmount"]
            .sum()
        )

        total_orders = (
            df["Invoice"].nunique()
            if "Invoice" in df.columns
            else len(df)
        )

        total_customers = (
            df["Customer ID"].nunique()
            if "Customer ID" in df.columns
            else 0
        )

        aov = (
            total_sales / total_orders
            if total_orders > 0
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "💰 Total Sales",
                f"₹{total_sales:,.2f}"
            )

        with col2:

            st.metric(
                "👥 Customers",
                f"{total_customers:,}"
            )

        with col3:

            st.metric(
                "🧾 Orders",
                f"{total_orders:,}"
            )

        with col4:

            st.metric(
                "🛒 Average Order Value",
                f"₹{aov:,.2f}"
            )

        st.divider()

        # ------------------------------------------
        # MONTHLY SALES
        # ------------------------------------------

        if "InvoiceDate" in df.columns:

            monthly_sales = (
                df.dropna(
                    subset=["InvoiceDate"]
                )
                .set_index("InvoiceDate")[
                    "TotalAmount"
                ]
                .resample("ME")
                .sum()
            )

            st.subheader(
                "📈 Monthly Sales Trend"
            )

            st.line_chart(
                monthly_sales
            )

        # ------------------------------------------
        # TOP COUNTRIES
        # ------------------------------------------

        if "Country" in df.columns:

            st.subheader(
                "🌍 Sales by Country"
            )

            country_sales = (
                df.groupby("Country")[
                    "TotalAmount"
                ]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
            )

            st.bar_chart(
                country_sales
            )

            st.dataframe(
                country_sales
                .reset_index(),
                use_container_width=True
            )

        # ------------------------------------------
        # TOP PRODUCTS
        # ------------------------------------------

        if "StockCode" in df.columns:

            st.subheader(
                "🏆 Top Products by Sales"
            )

            product_sales = (
                df.groupby("StockCode")[
                    "TotalAmount"
                ]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
            )

            st.bar_chart(
                product_sales
            )


# ==================================================
# CHURN PREDICTION
# ==================================================

elif page == "❤️ Churn Prediction":

    st.title(
        "❤️ Customer Churn Prediction"
    )

    st.write(
        "Enter Frequency and Monetary values "
        "to estimate customer churn probability."
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

    if st.button(
        "🔮 Predict Churn"
    ):

        if churn_model is None:

            st.error(
                "Churn model is not available."
            )

        else:

            input_data = np.array(
                [
                    [
                        frequency,
                        monetary
                    ]
                ]
            )

            probability = (
                churn_model
                .predict_proba(
                    input_data
                )[0, 1]
            )

            prediction = (
                churn_model
                .predict(
                    input_data
                )[0]
            )

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

            # Risk category
            if probability >= 0.70:

                risk = "High Risk"

                st.error(
                    "🔴 High Risk — "
                    "Customer is likely to churn."
                )

            elif probability >= 0.40:

                risk = "Medium Risk"

                st.warning(
                    "🟡 Medium Risk — "
                    "Customer needs attention."
                )

            else:

                risk = "Low Risk"

                st.success(
                    "🟢 Low Risk — "
                    "Customer is likely to stay."
                )

            st.write(
                f"**Risk Category:** {risk}"
            )

            st.write(
                f"**Model Prediction:** "
                f"{'Churn' if prediction == 1 else 'Stay'}"
            )

    # ----------------------------------------------
    # EXISTING CHURN ANALYSIS
    # ----------------------------------------------

    if churn_df is not None:

        st.divider()

        st.subheader(
            "📊 Customer Churn Risk Distribution"
        )

        if "Risk_Category" in churn_df.columns:

            risk_counts = (
                churn_df[
                    "Risk_Category"
                ]
                .value_counts()
            )

            st.bar_chart(
                risk_counts
            )

            st.subheader(
                "🔴 High-Risk Customers"
            )

            if "Churn_Probability" in churn_df.columns:

                high_risk = (
                    churn_df[
                        churn_df[
                            "Risk_Category"
                        ] == "High Risk"
                    ]
                    .sort_values(
                        "Churn_Probability",
                        ascending=False
                    )
                )

                st.dataframe(
                    high_risk.head(20),
                    use_container_width=True
                )


# ==================================================
# CUSTOMER SEGMENTATION
# ==================================================

elif page == "👥 Customer Segmentation":

    st.title(
        "👥 Customer Segmentation"
    )

    st.write(
        "Identify customer segments using "
        "trained K-Means clustering."
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

    if st.button(
        "🔍 Identify Customer Segment"
    ):

        if (
            kmeans_model is None
            or rfm_scaler is None
        ):

            st.error(
                "Segmentation model or scaler "
                "is not available."
            )

        else:

            input_data = np.array(
                [
                    [
                        recency,
                        frequency,
                        monetary
                    ]
                ]
            )

            scaled_input = (
                rfm_scaler
                .transform(input_data)
            )

            cluster = (
                kmeans_model
                .predict(
                    scaled_input
                )[0]
            )

            st.success(
                f"Customer belongs to "
                f"Segment / Cluster: {cluster}"
            )

    # ----------------------------------------------
    # EXISTING SEGMENTS
    # ----------------------------------------------

    if segments_df is not None:

        st.divider()

        st.subheader(
            "📊 Customer Segment Distribution"
        )

        possible_cluster_columns = [
            "Cluster",
            "cluster",
            "Segment",
            "segment"
        ]

        cluster_column = None

        for col in possible_cluster_columns:

            if col in segments_df.columns:

                cluster_column = col
                break

        if cluster_column:

            segment_counts = (
                segments_df[
                    cluster_column
                ]
                .value_counts()
            )

            st.bar_chart(
                segment_counts
            )

        st.subheader(
            "👥 Customer Segments"
        )

        st.dataframe(
            segments_df.head(20),
            use_container_width=True
        )


# ==================================================
# SALES FORECASTING
# ==================================================

elif page == "📈 Sales Forecasting":

    st.title(
        "📈 Sales Forecasting"
    )

    st.write(
        "Compare LSTM, Prophet and Hybrid "
        "forecasting results for demand planning."
    )

    st.divider()

    # ----------------------------------------------
    # LSTM
    # ----------------------------------------------

    if lstm_forecast_df is not None:

        st.subheader(
            "🔮 Next 30 Days — LSTM Forecast"
        )

        lstm_display = (
            lstm_forecast_df.copy()
        )

        lstm_display["ds"] = (
            pd.to_datetime(
                lstm_display["ds"]
            )
        )

        if "PredictedSales" in lstm_display.columns:

            st.line_chart(
                lstm_display
                .set_index("ds")[
                    "PredictedSales"
                ]
            )

        st.dataframe(
            lstm_display,
            use_container_width=True
        )

    else:

        st.warning(
            "LSTM_Future_Forecast.csv not found."
        )

    # ----------------------------------------------
    # HYBRID
    # ----------------------------------------------

    if hybrid_df is not None:

        st.divider()

        st.subheader(
            "🤖 Hybrid Forecast — Prophet + LSTM"
        )

        hybrid_display = (
            hybrid_df.copy()
        )

        hybrid_display["ds"] = (
            pd.to_datetime(
                hybrid_display["ds"]
            )
        )

        chart_columns = []

        if "yhat" in hybrid_display.columns:

            chart_columns.append(
                "yhat"
            )

        if "PredictedSales" in hybrid_display.columns:

            chart_columns.append(
                "PredictedSales"
            )

        if "HybridPrediction" in hybrid_display.columns:

            chart_columns.append(
                "HybridPrediction"
            )

        if chart_columns:

            st.line_chart(
                hybrid_display
                .set_index("ds")[
                    chart_columns
                ]
            )

        st.dataframe(
            hybrid_display[
                [
                    col
                    for col in [
                        "ds",
                        "yhat",
                        "PredictedSales",
                        "HybridPrediction"
                    ]
                    if col in hybrid_display.columns
                ]
            ],
            use_container_width=True
        )

    else:

        st.info(
            "Hybrid forecast CSV not found. "
            "If you saved Prophet_Daily_Future_Forecast.csv "
            "and LSTM_Future_Forecast.csv, the hybrid "
            "forecast will be generated automatically."
        )

    # ----------------------------------------------
    # EXISTING PROPHET FORECAST
    # ----------------------------------------------

    if sales_forecast_df is not None:

        st.divider()

        st.subheader(
            "📈 Prophet Historical Forecast"
        )

        prophet_display = (
            sales_forecast_df.copy()
        )

        if "ds" in prophet_display.columns:

            prophet_display["ds"] = (
                pd.to_datetime(
                    prophet_display["ds"]
                )
            )

        st.dataframe(
            prophet_display.head(20),
            use_container_width=True
        )

    # ----------------------------------------------
    # PROPHET FUTURE FORECAST
    # ----------------------------------------------

    if prophet_model is not None:

        st.divider()

        months = st.slider(
            "Prophet Forecast Horizon (months)",
            min_value=1,
            max_value=12,
            value=6
        )

        if st.button(
            "🔮 Generate Prophet Forecast"
        ):

            future = (
                prophet_model
                .make_future_dataframe(
                    periods=months,
                    freq="ME"
                )
            )

            forecast = (
                prophet_model
                .predict(future)
            )

            forecast_future = (
                forecast.tail(months)
            )

            st.subheader(
                "Prophet Future Forecast"
            )

            forecast_columns = [
                "ds",
                "yhat",
                "yhat_lower",
                "yhat_upper"
            ]

            st.dataframe(
                forecast_future[
                    forecast_columns
                ],
                use_container_width=True
            )

            st.line_chart(
                forecast_future
                .set_index("ds")[
                    "yhat"
                ]
            )


# ==================================================
# INVENTORY OPTIMIZATION
# ==================================================

elif page == "📦 Inventory Optimization":

    st.title(
        "📦 Inventory Optimization"
    )

    st.write(
        "Demand-based inventory recommendations "
        "using demand variability, safety stock "
        "and reorder point analysis."
    )

    st.divider()

    if inventory_df is None:

        st.error(
            "Inventory_Recommendations.csv not found."
        )

    else:

        required_columns = [
            "Recommendation",
            "RecommendedReorderQty",
            "StockCode"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in inventory_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing inventory columns: "
                + ", ".join(missing_columns)
            )

        else:

            total_products = (
                len(inventory_df)
            )

            reorder_products = (
                inventory_df[
                    "Recommendation"
                ]
                .eq("REORDER")
                .sum()
            )

            total_reorder_qty = (
                inventory_df[
                    "RecommendedReorderQty"
                ]
                .sum()
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "📦 Total Products",
                    f"{total_products:,}"
                )

            with col2:

                st.metric(
                    "🔴 Products to Reorder",
                    f"{reorder_products:,}"
                )

            with col3:

                st.metric(
                    "📊 Recommended Reorder Qty",
                    f"{total_reorder_qty:,.0f}"
                )

            st.divider()

            st.subheader(
                "🔴 Top Products Requiring Reorder"
            )

            priority_products = (
                inventory_df[
                    inventory_df[
                        "Recommendation"
                    ] == "REORDER"
                ]
                .sort_values(
                    "RecommendedReorderQty",
                    ascending=False
                )
                .head(20)
            )

            st.dataframe(
                priority_products,
                use_container_width=True
            )

            st.divider()

            st.subheader(
                "📊 Top 10 Reorder Recommendations"
            )

            chart_data = (
                priority_products
                .head(10)
                .set_index("StockCode")[
                    "RecommendedReorderQty"
                ]
            )

            st.bar_chart(
                chart_data
            )

            st.divider()

            csv = (
                inventory_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "📥 Download Inventory Recommendations",
                csv,
                "Inventory_Recommendations.csv",
                "text/csv"
            )


# ==================================================
# BUSINESS INSIGHTS
# ==================================================

elif page == "💡 Business Insights":

    st.title(
        "💡 Business Insights & Recommendations"
    )

    st.write(
        "Actionable business insights generated "
        "from customer analytics, churn prediction, "
        "sales forecasting and inventory optimization."
    )

    st.divider()

    # ----------------------------------------------
    # CHURN INSIGHTS
    # ----------------------------------------------

    if churn_df is not None:

        if "Risk_Category" in churn_df.columns:

            high_risk_count = (
                churn_df[
                    "Risk_Category"
                ]
                .eq("High Risk")
                .sum()
            )

            medium_risk_count = (
                churn_df[
                    "Risk_Category"
                ]
                .eq("Medium Risk")
                .sum()
            )

            low_risk_count = (
                churn_df[
                    "Risk_Category"
                ]
                .eq("Low Risk")
                .sum()
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "🔴 High-Risk Customers",
                    f"{high_risk_count:,}"
                )

            with col2:

                st.metric(
                    "🟡 Medium-Risk Customers",
                    f"{medium_risk_count:,}"
                )

            with col3:

                st.metric(
                    "🟢 Low-Risk Customers",
                    f"{low_risk_count:,}"
                )

            st.subheader(
                "📊 Churn Risk Distribution"
            )

            risk_distribution = (
                churn_df[
                    "Risk_Category"
                ]
                .value_counts()
            )

            st.bar_chart(
                risk_distribution
            )

            st.subheader(
                "🎯 Recommended Churn Actions"
            )

            st.write(
                "• Prioritize high-risk customers "
                "for retention campaigns."
            )

            st.write(
                "• Monitor purchase frequency "
                "as a customer engagement indicator."
            )

            st.write(
                "• Offer personalized discounts "
                "to customers showing reduced engagement."
            )

    # ----------------------------------------------
    # FORECAST INSIGHTS
    # ----------------------------------------------

    if lstm_forecast_df is not None:

        if (
            "PredictedSales"
            in lstm_forecast_df.columns
        ):

            average_forecast = (
                lstm_forecast_df[
                    "PredictedSales"
                ]
                .mean()
            )

            maximum_forecast = (
                lstm_forecast_df[
                    "PredictedSales"
                ]
                .max()
            )

            minimum_forecast = (
                lstm_forecast_df[
                    "PredictedSales"
                ]
                .min()
            )

            st.divider()

            st.subheader(
                "📈 Forecast Insights"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Average Daily Forecast",
                    f"₹{average_forecast:,.2f}"
                )

            with col2:

                st.metric(
                    "Maximum Forecast",
                    f"₹{maximum_forecast:,.2f}"
                )

            with col3:

                st.metric(
                    "Minimum Forecast",
                    f"₹{minimum_forecast:,.2f}"
                )

            st.write(
                "The LSTM model provides a 30-day "
                "sales forecast that can support "
                "inventory and sales planning."
            )

    # ----------------------------------------------
    # INVENTORY INSIGHTS
    # ----------------------------------------------

    if inventory_df is not None:

        if (
            "Recommendation"
            in inventory_df.columns
        ):

            reorder_count = (
                inventory_df[
                    "Recommendation"
                ]
                .eq("REORDER")
                .sum()
            )

            st.divider()

            st.subheader(
                "📦 Inventory Insight"
            )

            st.metric(
                "Products Requiring Reorder",
                f"{reorder_count:,}"
            )

            st.write(
                "Products with high demand or "
                "insufficient current stock should "
                "be prioritized for replenishment."
            )

    # ----------------------------------------------
    # CUSTOMER SEGMENTATION INSIGHT
    # ----------------------------------------------

    if segments_df is not None:

        st.divider()

        st.subheader(
            "👥 Customer Segmentation Insight"
        )

        possible_cluster_columns = [
            "Cluster",
            "cluster",
            "Segment",
            "segment"
        ]

        cluster_column = None

        for col in possible_cluster_columns:

            if col in segments_df.columns:

                cluster_column = col
                break

        if cluster_column:

            segment_counts = (
                segments_df[
                    cluster_column
                ]
                .value_counts()
            )

            largest_segment = (
                segment_counts
                .idxmax()
            )

            largest_segment_count = (
                segment_counts
                .max()
            )

            st.metric(
                "Largest Customer Segment",
                str(largest_segment)
            )

            st.write(
                f"Cluster {largest_segment} "
                f"contains approximately "
                f"{largest_segment_count:,} customers."
            )

    # ----------------------------------------------
    # FINAL RECOMMENDATIONS
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "🎯 Business Recommendations"
    )

    st.write(
        "1. Focus retention campaigns on "
        "high-risk customers."
    )

    st.write(
        "2. Monitor customer purchase frequency "
        "regularly."
    )

    st.write(
        "3. Use customer segments for "
        "targeted marketing."
    )

    st.write(
        "4. Use LSTM, Prophet and Hybrid forecasts "
        "for demand planning."
    )

    st.write(
        "5. Prioritize products flagged for "
        "inventory reorder."
    )

    st.write(
        "6. Combine churn and segmentation insights "
        "for personalized customer campaigns."
    )

    st.success(
        "RetailPulse converts analytics and ML "
        "outputs into actionable retail decisions. 🚀"
    )