import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def run_clv_dashboard():
    st.title("📊 CLV Dashboard (CSV, Excel, Google Sheets Supported)")

    # ========== FILE UPLOAD OPTIONS (now under title) ==========
    st.header("📤 Upload Options")
    upload_type = st.radio("Select Upload Type:", ["CSV", "Excel", "Google Sheet"])

    df = None

    if upload_type == "CSV":
        file = st.file_uploader("Upload CSV file", type=["csv"])
        if file:
            try:
                df = pd.read_csv(file, encoding="utf-8", on_bad_lines='skip')
                st.success("✅ CSV uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading CSV: {e}")

    elif upload_type == "Excel":
        file = st.file_uploader("Upload Excel (.xlsx) file", type=["xlsx"])
        if file:
            try:
                df = pd.read_excel(file)
                st.success("✅ Excel file uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading Excel: {e}")

    elif upload_type == "Google Sheet":
        sheet_url = st.text_input("Paste Google Sheets shareable link")
        if sheet_url:
            if "/edit" in sheet_url:
                sheet_url = sheet_url.replace("/edit", "/export?format=csv")
            elif "?usp=sharing" in sheet_url:
                sheet_url = sheet_url.replace("?usp=sharing", "/export?format=csv")
            try:
                df = pd.read_csv(sheet_url, on_bad_lines='skip')
                st.success("✅ Data loaded from Google Sheets!")
            except Exception as e:
                st.error(f"❌ Failed to load from Google Sheets: {e}")

    # ========== PROCESSING CLV ========== #
    if df is not None:
        try:
            # Date Conversion
            if "Dt_Customer" in df.columns:
                df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors='coerce')
            else:
                st.error("❌ 'Dt_Customer' column is required.")
                st.stop()

            # Age & Tenure
            df["Age"] = datetime.now().year - df["Year_Birth"] if "Year_Birth" in df.columns else None
            df["Customer_Tenure"] = (datetime.now() - df["Dt_Customer"]).dt.days
            df["Tenure_Years"] = df["Customer_Tenure"] / 365

            # Spending & Purchases
            spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
            for col in spend_cols:
                if col not in df.columns:
                    df[col] = 0
            df["Total_Spending"] = df[spend_cols].sum(axis=1)

            purchase_cols = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
            for col in purchase_cols:
                if col not in df.columns:
                    df[col] = 0
            df["Purchase_Frequency"] = df[purchase_cols].sum(axis=1)

            # CLV Calculation
            df["Profit_Margin"] = df["Total_Spending"] * 0.3
            df["CLV"] = (df["Profit_Margin"] * df["Purchase_Frequency"] * df["Tenure_Years"]).round(2)

            # ========== FILTERS ==========
            st.sidebar.header("🔍 Filters")
            gender_opt = df["Gender"].dropna().unique().tolist() if "Gender" in df.columns else []
            marital_opt = df["Marital_Status"].dropna().unique().tolist() if "Marital_Status" in df.columns else []
            edu_opt = df["Education"].dropna().unique().tolist() if "Education" in df.columns else []

            gender_filter = st.sidebar.multiselect("Gender", gender_opt, default=gender_opt)
            marital_filter = st.sidebar.multiselect("Marital Status", marital_opt, default=marital_opt)
            edu_filter = st.sidebar.multiselect("Education", edu_opt, default=edu_opt)

            if "Gender" in df.columns:
                df = df[df["Gender"].isin(gender_filter)]
            if "Marital_Status" in df.columns:
                df = df[df["Marital_Status"].isin(marital_filter)]
            if "Education" in df.columns:
                df = df[df["Education"].isin(edu_filter)]

            # ========== CLV TABLE ==========
            st.subheader("📈 CLV Summary Table")
            show_cols = ["ID", "Age", "Total_Spending", "Purchase_Frequency", "Tenure_Years", "CLV"]
            show_cols = [col for col in show_cols if col in df.columns]
            st.dataframe(df[show_cols].sort_values(by="CLV", ascending=False).reset_index(drop=True))

            # ========== DOWNLOAD ==========
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download CLV Results CSV", csv, "clv_results.csv", "text/csv")

            # ========== PIE CHART ==========
            st.subheader("🥧 CLV Distribution Pie Chart")
            pie_options = ["Age Group"]
            if "Country" in df.columns:
                pie_options.append("Region")

            pie_by = st.selectbox("Group CLV By", pie_options)

            if pie_by == "Age Group":
                bins = [0, 30, 40, 50, 60, 100]
                labels = ['<30', '30-40', '40-50', '50-60', '60+']
                df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
                clv_by_age = df.groupby("AgeGroup")["CLV"].sum().dropna()
                fig, ax = plt.subplots(figsize=(6, 6))
                clv_by_age.plot.pie(autopct='%1.1f%%', ylabel='', ax=ax)
                ax.set_title("CLV by Age Group")
                st.pyplot(fig)

            elif pie_by == "Region":
                clv_by_region = df.groupby("Country")["CLV"].sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(6, 6))
                clv_by_region.plot.pie(autopct='%1.1f%%', ylabel='', ax=ax)
                ax.set_title("CLV by Region (Top 10)")
                st.pyplot(fig)

            # ========== BAR CHART ==========
            st.subheader("📊 Top Customers by CLV")
            top_n = st.slider("Select Top N", 5, 50, 10)
            if "ID" in df.columns:
                top_clv = df.sort_values(by="CLV", ascending=False).head(top_n)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(top_clv["ID"].astype(str), top_clv["CLV"], color='skyblue')
                ax.set_title("Top Customers by CLV")
                ax.set_ylabel("CLV")
                ax.set_xlabel("Customer ID")
                plt.xticks(rotation=45)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"🚨 Error processing file: {e}")
