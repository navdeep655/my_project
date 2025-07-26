import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def run_clv():

    # Custom CSS
    st.markdown("""
        <style>
        h1, h2, h3 {
            color: #2E7D32;
        }
        .stMetric {
            background-color: #E8F5E9;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stDownloadButton>button {
            background-color: #388E3C;
            color: white;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📈 Customer Lifetime Value (CLV) Prediction")

    uploaded_file = st.file_uploader("📤 Upload your customer_data.csv file", type=["csv"], key='customer')

    @st.cache_data
    def load_data(uploaded_file):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
        else:
            df = pd.read_csv("customer_data.csv")
        return df

    raw_df = load_data(uploaded_file)
    df = raw_df.copy()

    # Feature Engineering
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    current_year = datetime.now().year
    df["Age"] = current_year - df["Year_Birth"]
    df["Customer_Tenure"] = (datetime.now() - df["Dt_Customer"]).dt.days
    df["Tenure_Years"] = df["Customer_Tenure"] / 365

    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spending"] = df[spend_cols].sum(axis=1)

    purchase_cols = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    df["Purchase_Frequency"] = df[purchase_cols].sum(axis=1)

    df["Profit_Margin"] = df["Total_Spending"] * 0.3
    df["CLV"] = df["Profit_Margin"] * df["Purchase_Frequency"] * df["Tenure_Years"]
    df["CLV"] = df["CLV"].round(2)
    df.dropna(inplace=True)

    def train_model(df):
        features = [
            'Age', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntGoldProds',
            'NumDealsPurchases', 'AcceptedCmp1', 'AcceptedCmp5', 'Response',
            'NumCatalogPurchases', 'Customer_Tenure', 'Tenure_Years',
            'Total_Spending', 'Purchase_Frequency', 'Profit_Margin'
        ]
        X = df[features]
        y = df["CLV"]

        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X, y)

        joblib.dump(model, "xgb_model.pkl")
        joblib.dump(features, "features.pkl")

        return model, features

    def user_input(df):
        st.sidebar.header("🧾 Enter Customer Details")

        Age = st.sidebar.slider("Age", 18, 100, int(df["Age"].median()))
        Income = st.sidebar.slider("Income", 0, int(df["Income"].max()), int(df["Income"].median()))
        Kidhome = st.sidebar.selectbox("Kidhome", [0,1,2,3])
        Teenhome = st.sidebar.selectbox("Teenhome", [0,1,2,3])
        Recency = st.sidebar.slider("Recency (days since last purchase)", 0, 100, int(df["Recency"].median()))
        MntGoldProds = st.sidebar.slider("Amount spent on Gold Products", 0, int(df["MntGoldProds"].max()), int(df["MntGoldProds"].median()))
        NumDealsPurchases = st.sidebar.slider("Number of Deal Purchases", 0, int(df["NumDealsPurchases"].max()), int(df["NumDealsPurchases"].median()))
        AcceptedCmp1 = st.sidebar.selectbox("Accepted Campaign 1", [0,1])
        AcceptedCmp5 = st.sidebar.selectbox("Accepted Campaign 5", [0,1])
        Response = st.sidebar.selectbox("Response", [0,1])
        NumCatalogPurchases = st.sidebar.slider("Number of Catalog Purchases", 0, int(df["NumCatalogPurchases"].max()), int(df["NumCatalogPurchases"].median()))
        Customer_Tenure = st.sidebar.slider("Customer Tenure (days)", 0, int(df["Customer_Tenure"].max()), int(df["Customer_Tenure"].median()))

        Tenure_Years = Customer_Tenure / 365
        Total_Spending = MntGoldProds
        Purchase_Frequency = NumDealsPurchases + NumCatalogPurchases
        Profit_Margin = Total_Spending * 0.3

        data = {
            "Age": Age,
            "Income": Income,
            "Kidhome": Kidhome,
            "Teenhome": Teenhome,
            "Recency": Recency,
            "MntGoldProds": MntGoldProds,
            "NumDealsPurchases": NumDealsPurchases,
            "AcceptedCmp1": AcceptedCmp1,
            "AcceptedCmp5": AcceptedCmp5,
            "Response": Response,
            "NumCatalogPurchases": NumCatalogPurchases,
            "Customer_Tenure": Customer_Tenure,
            "Tenure_Years": Tenure_Years,
            "Total_Spending": Total_Spending,
            "Purchase_Frequency": Purchase_Frequency,
            "Profit_Margin": Profit_Margin
        }

        return pd.DataFrame([data])

    # Model load/retrain
    if st.sidebar.button("🔁 Retrain Model"):
        with st.spinner("Training model..."):
            model, features = train_model(df)
            st.success("Model retrained and saved!")
    elif not os.path.exists("xgb_model.pkl") or not os.path.exists("features.pkl"):
        with st.spinner("Training model..."):
            model, features = train_model(df)
            st.success("Model trained and saved!")
    else:
        model = joblib.load("xgb_model.pkl")
        features = joblib.load("features.pkl")

    # Show dataset
    with st.expander("📊 View Dataset"):
        st.write(raw_df.head())

    # Predict
    input_df = user_input(df)
    predict_df = input_df[features]

    if st.sidebar.button("🎯 Predict CLV"):
        prediction = model.predict(predict_df)[0]

        profit = input_df["Profit_Margin"].values[0]
        freq = input_df["Purchase_Frequency"].values[0]
        tenure = input_df["Tenure_Years"].values[0]
        manual_clv = profit * freq * tenure
        annualized_clv = profit * (freq / tenure)

        st.markdown("## 💡 CLV Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📊 Predicted CLV (Model)", value=f"₹ {prediction:,.2f}")
        with col2:
            st.metric(label="🧮 Manual CLV (Formula)", value=f"₹ {manual_clv:,.2f}")
        with col3:
            st.metric(label="📆 1-Year CLV Estimate", value=f"₹ {annualized_clv:,.2f}")

        st.markdown("---")
        st.markdown("### 📌 CLV Components")
        st.markdown(f"""
        - **Profit Margin:** ₹ {profit:,.2f}  
        - **Purchase Frequency:** {freq}  
        - **Tenure (Years):** {tenure:.2f}
        """)

        st.markdown("### 🧠 Explanation")
        st.info("""
        **Predicted CLV (Model):**  
        This is calculated using an XGBoost ML model trained on 16+ customer behavior features.

        **Manual CLV (Formula):**  
        This uses a basic formula: `Profit × Frequency × Tenure`, which gives a baseline.

        **Why are they different?**  
        The model has learned from historical data that certain customer behaviors (like high income, frequent purchases, campaign responses, etc.) lead to higher long-term value. So it may predict a higher CLV even if the manual value is lower.
        """)

        # Save & download
        input_df["Model_Predicted_CLV"] = prediction
        input_df["Manual_CLV"] = manual_clv
        input_df["Annualized_CLV"] = annualized_clv

        st.download_button("⬇️ Download Result", input_df.to_csv(index=False), "clv_result.csv")
