# RetailPulse – AI-Powered Retail Analytics & Forecasting Platform

## 📌 Project Overview

**RetailPulse** is an end-to-end retail analytics and machine learning project designed to transform raw retail transaction data into actionable business insights.

The project combines **data cleaning, feature engineering, exploratory data analysis, RFM customer analysis, machine-learning-based customer segmentation, sales forecasting, Power BI visualization, and an interactive Streamlit application**.

The goal is to help businesses understand **what happened, why it happened, which customers matter most, and what may happen next**.

---

## 🎯 Objectives

* Clean and prepare raw retail transaction data.
* Perform exploratory data analysis to identify sales patterns.
* Analyze customer purchasing behavior using **RFM (Recency, Frequency, Monetary)** analysis.
* Segment customers using **K-Means clustering**.
* Forecast future sales using time-series forecasting techniques.
* Build an interactive **Power BI dashboard** for business intelligence.
* Develop a **Streamlit application** to provide an interactive interface for the analytics/ML workflow.
* Generate insights that can support customer retention, sales planning, and business decision-making.

---

## 🔄 Project Workflow

```text
Raw Retail Data
       ↓
Data Loading & Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
RFM Customer Analysis
       ↓
Customer Segmentation
(K-Means Clustering)
       ↓
Sales Forecasting
       ↓
Power BI Dashboard
       ↓
Streamlit Application
```

---

## 📂 Project Structure

```text
RetailPulse/
│
├── 01_Data_Loading_Cleaning(5).ipynb
├── 02_Feature_Engineering(3).ipynb
├── 03_Exploratory_Data_Analysis(1).ipynb
├── 04_Customer_RFM_Analysis(2).ipynb
├── 05_Customer_Segmentation(2).ipynb
├── 06_Sales_Forecasting(1).ipynb
│
├── Customer_RFM.csv
│
├── Dashboard/
│   └── RetailPulse_Dashboard.pbix
│
├── Images/
│   └── RetailPulse_Dashboard.png
│
├── app.py
│
└── README.md
```

> File/folder names can be updated if the final GitHub structure uses different names.

---

## 🧹 1. Data Loading & Cleaning

The raw retail transaction dataset was loaded and prepared for analysis.

Major preprocessing steps included:

* Handling missing values
* Removing/handling invalid records
* Checking duplicate transactions
* Correcting data types
* Handling cancelled/invalid transactions
* Creating a reliable transaction-level dataset
* Preparing the cleaned data for downstream analysis

---

## ⚙️ 2. Feature Engineering

Additional features were created from the transaction data to support deeper analysis.

Examples include:

* Total Amount
* Year
* Month
* Month Name
* Day
* Weekday
* Hour
* Weekend indicator
* Quarter
* Time of Day

These features helped identify **temporal sales patterns and customer behavior**.

---

## 📊 3. Exploratory Data Analysis

EDA was performed to understand the overall sales and customer behavior.

The analysis includes:

* Sales by country
* Top customers
* Hourly sales patterns
* Weekday sales patterns
* Monthly sales trends
* Revenue analysis
* Average Order Value (AOV)
* Customer purchasing behavior

### Key Finding

The analysis identified differences in purchasing behavior across **countries, customers, weekdays, and time periods**, providing the foundation for customer segmentation and forecasting.

---

## 👥 4. RFM Customer Analysis

RFM analysis was used to evaluate customer value based on three dimensions:

| Metric        | Meaning                             |
| ------------- | ----------------------------------- |
| **Recency**   | How recently a customer purchased   |
| **Frequency** | How frequently a customer purchased |
| **Monetary**  | How much a customer spent           |

The resulting RFM dataset was used as an input for customer segmentation.

---

## 🤖 5. Customer Segmentation

**K-Means clustering** was applied to the RFM features to identify groups of customers with similar purchasing behavior.

The segmentation helps identify customer groups such as:

* High-value customers
* Loyal/active customers
* Regular customers
* Customers requiring re-engagement

The clustering results can support targeted marketing and customer-retention strategies.

### Model Evaluation

Cluster quality was evaluated using clustering evaluation techniques such as the **Silhouette Score**.

---

## 📈 6. Sales Forecasting

The project also includes a dedicated sales forecasting workflow.

The forecasting component analyzes historical sales patterns to estimate future sales trends.

The forecasting workflow is used to support:

* Future sales planning
* Demand estimation
* Inventory planning
* Business decision-making

The detailed forecasting implementation is available in:

`06_Sales_Forecasting(1).ipynb`

---

## 🚀 Live Streamlit Application

👉 **[Launch RetailPulse App](https://kzmk49dfonbbdhekwbz3pr.streamlit.app/)**

The RetailPulse Streamlit application provides an interactive interface for the trained machine-learning models.

### Available Features

* ❤️ **Customer Churn Prediction** — predicts whether a customer is likely to churn based on customer purchase behavior.
* 👥 **Customer Segmentation** — uses RFM values and a trained K-Means model to identify the customer's cluster.
* 📈 **Sales Forecasting** — uses a trained Prophet model to forecast future monthly sales for a user-selected forecast horizon.

The application provides a user-friendly way to interact with the project's machine-learning models without running the notebooks manually.


## 📊 7. Power BI Dashboard

An interactive **Power BI dashboard** was created to present the project's business insights in an easy-to-understand format.

The dashboard provides visual analysis of:

* Revenue
* Orders
* Customers
* Average Order Value
* Sales trends
* Customer insights
* Sales forecasting
* Other key business metrics

The dashboard acts as the **business intelligence and visualization layer** of RetailPulse.

---

## 🚀 8. Streamlit Application

A **Streamlit application** was developed to provide an interactive interface for the RetailPulse analytics/ML workflow.

The application makes the project easier to demonstrate and allows users to interact with the implemented analysis/models through a web-based interface.

### Application

The Streamlit application is implemented in:

```text
app.py
```

---

## 🛠️ Technologies Used

### Programming & Data Science

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

### Machine Learning

* K-Means Clustering
* RFM-based Customer Segmentation
* Time-Series Forecasting

### Visualization & BI

* Power BI
* Streamlit
* Matplotlib

### Development Tools

* Jupyter Notebook
* Git
* GitHub

---

## 💡 Business Value

RetailPulse can help a retail business:

* Identify valuable customer segments
* Understand customer purchasing behavior
* Monitor sales performance
* Identify important sales patterns
* Estimate future sales
* Improve customer targeting
* Support inventory and sales planning
* Make data-driven business decisions

---

## ⭐ Key Project Highlights

* End-to-end retail analytics workflow
* RFM-based customer intelligence
* Machine-learning-based customer segmentation
* Sales forecasting
* Interactive Power BI dashboard
* Interactive Streamlit application
* Business-focused insights from transactional data

---

## 📌 Conclusion

**RetailPulse** combines traditional business intelligence with machine learning and interactive application development to create an end-to-end retail analytics solution.

The project demonstrates how raw transaction data can be transformed into:

**Data → Insights → Customer Segments → Forecasts → Business Decisions**

---

## 👩‍💻 Author

**Priya Rani**

Data Science & Analytics | Python | SQL | Power BI | Machine Learning
