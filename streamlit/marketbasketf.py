import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import os

def run_marketbasket():
    st.title("🛒 Market Basket Analysis")

    # CSS styling
    st.markdown(
        """
        <style>
        div[role="radiogroup"] > label > div {
            font-size: 20px;
            font-weight: 600;
            color: #1f77b4;
        }
        .stRadio > div {
            gap: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def find_column(cols, keywords):
        for col in cols:
            for kw in keywords:
                norm_col = col.lower().replace("_", " ").replace("-", " ")
                norm_kw = kw.lower().replace("_", " ").replace("-", " ")
                if norm_kw in norm_col:
                    return col
        return None

    uploaded_file = st.file_uploader("Upload your transaction CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(uploaded_file)

        cols = df.columns.tolist()
        invoice_col = find_column(cols, ["bill no", "billno", "invoice", "invoice no", "transaction id"])
        product_col = find_column(cols, ["item name", "item", "product", "product name"])

        if invoice_col is None or product_col is None:
            st.error("CSV must contain 'Bill No' and 'Item Name' columns (or equivalents).")
            st.stop()

        st.write(f"Detected Invoice column: `{invoice_col}`")
        st.write(f"Detected Product column: `{product_col}`")

        # Filter for frequent items
        item_counts = df[product_col].value_counts()
        common_items = item_counts[item_counts > 500].index
        df = df[df[product_col].isin(common_items)]

        basket = df.groupby([invoice_col, product_col]).size().unstack(fill_value=0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        # Generate frequent itemsets and rules
        freq_items = apriori(basket, min_support=0.02, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.4)

        # Filter only single-product pairs
        rules = rules[
            (rules['antecedents'].apply(len) == 1) &
            (rules['consequents'].apply(len) == 1)
        ]

        # Apply fixed filters
        rules = rules[
            (rules['support'] >= 0.02) &
            (rules['confidence'] >= 0.4) &
            (rules['lift'] >= 1.0)
        ]

        # Convert to displayable format
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: next(iter(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: next(iter(x)))

        st.write(f"### Product Pairs Bought Together ({len(rules)})")
        st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']])

        # Download button
        csv = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Product Pairs CSV",
            data=csv,
            file_name='product_pairs.csv',
            mime='text/csv'
        )

    else:
        st.info("Please upload a CSV file with 'Bill No' and 'Item Name' columns.")
