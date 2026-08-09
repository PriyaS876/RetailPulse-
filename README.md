📊 RetailPulse

AI-Powered Customer Analytics & Demand Forecasting Platform

RetailPulse is an end-to-end retail analytics and machine learning project that combines sales analytics, customer segmentation, churn prediction, demand forecasting, and inventory optimization in an interactive Streamlit dashboard.

The project follows a complete data-science workflow:

Data → Cleaning → Feature Engineering → EDA → Customer Analytics → ML Models → Forecasting → Inventory Recommendations → Streamlit Dashboard

🎯 Project Overview

Retail businesses need to understand customer behavior, anticipate future demand, identify customers at risk of churn, and maintain the right inventory levels.

RetailPulse addresses these problems through:

Sales and business KPI analysis

RFM-based customer analysis

K-Means customer segmentation

Churn prediction

Prophet sales forecasting

LSTM future forecasting

Hybrid Prophet + LSTM forecasting

Inventory reorder recommendations

Interactive Streamlit business dashboard

The original project workflow also defines deployment as part of the broader roadmap, while the current repository focuses on the implemented analytics, ML, forecasting, optimization, and dashboard layers.

✨ Key Features

Area

Implementation

Data Preparation

Cleaning, transformation and feature engineering

Sales Analytics

Revenue, orders, customers, AOV and sales trends

Customer Analytics

RFM analysis

Segmentation

K-Means clustering

Churn

Churn classification and risk analysis

Forecasting

Prophet + LSTM + Hybrid forecasting

Inventory

Safety stock, reorder point and reorder quantity recommendations

Dashboard

Multi-page Streamlit application

Business Insights

Actionable recommendations based on model outputs

🧠 Machine Learning & Analytics

1. Customer RFM Analysis

Customer behavior is represented using:

Recency – how recently a customer purchased

Frequency – how often a customer purchased

Monetary – how much a customer spent

The resulting RFM data is used for customer profiling and segmentation.

2. Customer Segmentation

A trained K-Means model is used to assign customers to segments based on RFM characteristics.

The Streamlit dashboard also provides an interactive way to enter RFM values and identify the corresponding customer cluster.

3. Churn Prediction

The application uses a trained churn classification model to estimate whether a customer is likely to churn.

The dashboard supports:

Purchase frequency input

Monetary value input

Churn prediction

Churn probability / risk interpretation

High-risk customer analysis when prediction data is available

4. Sales Forecasting

RetailPulse contains multiple forecasting approaches:

Prophet

A time-series model used for sales forecasting.

LSTM

A neural-network-based time-series model used for future sales prediction.

Hybrid Forecast

Prophet and LSTM predictions are combined to provide a hybrid forecast.

The current forecasting output includes a 30-day future prediction.

5. Inventory Optimization

Inventory recommendations are generated using demand statistics such as:

Total quantity

Average daily demand

Demand standard deviation

Safety stock

Reorder point

Current stock

Recommended reorder quantity

Reorder recommendation

This converts forecasting output into an operational inventory decision.

📊 Streamlit Dashboard

The dashboard contains the following sections:

🏠 Home

Provides an overview of the platform and key retail KPIs.

📊 Sales Analytics

Displays sales performance, trends and business-level metrics.

❤️ Churn Prediction

Allows interactive churn prediction and displays customer risk analysis.

👥 Customer Segmentation

Provides RFM-based customer segmentation and interactive cluster prediction.

📈 Sales Forecasting

Displays LSTM future forecasts and Prophet forecasting results, with support for comparing forecasting outputs.

📦 Inventory Optimization

Displays products requiring replenishment and recommended reorder quantities.

💡 Business Insights

Converts analytical and ML outputs into business recommendations.

🏗️ Project Architecture

                    Retail Sales Dataset
                            │
                            ▼
                 Data Loading & Cleaning
                            │
                            ▼
                  Feature Engineering
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
           EDA          RFM Analysis    Time Series
             │              │              │
             │              ▼              ├─────────────┐
             │        K-Means              ▼             ▼
             │       Segmentation       Prophet        LSTM
             │                             │             │
             │                             └──────┬──────┘
             │                                    ▼
             │                              Hybrid Forecast
             │                                    │
             ├──────────────┐                     ▼
             ▼              ▼             Inventory Optimization
        Churn Model     Customer Data             │
             │              │                     ▼
             └──────────────┴────────────► Business Insights
                                                  │
                                                  ▼
                                        Streamlit Dashboard

📁 Repository Structure

RetailPulse/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── Notebooks/
│   ├── 01_Data_Loading_Cleaning.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Customer_RFM_Analysis.ipynb
│   ├── 05_Customer_Segmentation.ipynb
│   ├── 06_Sales_Forecasting.ipynb
│   ├── 07_Churn_Prediction.ipynb
│   ├── 08_LSTM_Forecasting.ipynb
│   ├── 09_Inventory_Optimization.ipynb
│   ├── 10_Hybrid_Forecasting.ipynb
│   │
│   ├── Retail_Cleaned.csv
│   ├── Retail_Feature_Engineered.csv
│   ├── Customer_RFM.csv
│   ├── Customer_Segments.csv
│   ├── Sales_Forecast.csv
│   ├── LSTM_Future_Forecast.csv
│   └── Inventory_Recommendations.csv
│
└── Models/
    ├── churn_model.pkl
    ├── kmeans_model.pkl
    ├── prophet_model.pkl
    └── rfm_scaler.pkl

Keep large/raw datasets and generated model artifacts out of GitHub when they are too large. Use .gitignore and Git LFS or external storage when appropriate.

🧪 Notebook Workflow

Notebook

Purpose

01_Data_Loading_Cleaning.ipynb

Load and clean retail data

02_Feature_Engineering.ipynb

Create analytical features

03_Exploratory_Data_Analysis.ipynb

Explore sales and customer patterns

04_Customer_RFM_Analysis.ipynb

Build RFM customer profiles

05_Customer_Segmentation.ipynb

Train/evaluate customer clusters

06_Sales_Forecasting.ipynb

Prophet-based sales forecasting

07_Churn_Prediction.ipynb

Customer churn modeling

08_LSTM_Forecasting.ipynb

LSTM future forecasting

09_Inventory_Optimization.ipynb

Reorder and inventory recommendations

10_Hybrid_Forecasting.ipynb

Combine Prophet and LSTM forecasts

🛠️ Technology Stack

Programming & Data Science

Python

Pandas

NumPy

Scikit-learn

Machine Learning

XGBoost

K-Means

Joblib

Forecasting

Prophet

PyTorch / LSTM

Visualization

Matplotlib

Plotly

Streamlit

Development

Jupyter Notebook

Git

GitHub

🚀 Run Locally

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RetailPulse

2. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Start the Streamlit application

streamlit run app.py

The dashboard will open in your browser.

📌 Important Repository Notes

The Streamlit application expects trained model files and generated CSV outputs in the paths used by app.py.

Before publishing:

Verify every model file exists.

Verify every CSV referenced by app.py exists.

Remove machine-specific absolute paths such as C:\Users\....

Keep project-relative paths so the repository works on another computer.

Test the application from a fresh virtual environment.

Make sure no passwords, API keys or private credentials are committed.

📈 Current Project Status

Implemented

Data cleaning

Feature engineering

EDA

RFM analysis

Customer segmentation

Churn prediction

Prophet forecasting

LSTM forecasting

Hybrid forecasting

Inventory optimization

Streamlit dashboard

Business insights

Model and CSV outputs connected to the application

Optional Future Enhancements

These are not required for the current portfolio version, but can be added if targeting MLOps / ML Engineer roles:

Docker containerization

Kubernetes deployment

GitHub Actions CI/CD

MLflow experiment tracking

Evidently drift monitoring

Automated retraining pipeline

Cloud deployment

Prometheus/Grafana monitoring

💼 Portfolio Value

RetailPulse demonstrates an end-to-end understanding of:

Data Analytics + Machine Learning + Time-Series Forecasting + Customer Analytics + Business Intelligence + Streamlit Deployment

It is suitable for demonstrating skills relevant to:

Data Analyst

Data Scientist

Junior Data Scientist

Business/Data Analytics

Machine Learning Intern

Junior ML Engineer

For MLOps/ML Engineer positions, the optional production components listed above can further strengthen the project.

⚠️ Disclaimer

The project is a portfolio/learning implementation based on retail transaction data. Business impact targets from the original project specification should not be presented as measured results unless they have been experimentally validated.

👩‍💻 Author

Priya Rani

Data Science & Analytics | Machine Learning | Business Intelligence

⭐ Acknowledgements

This project was developed as an end-to-end Data Science & Analytics portfolio project inspired by the RetailPulse project specification and retail demand/customer analytics use case.
