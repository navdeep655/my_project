import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

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

    @st.cache_data(show_spinner=False)
    def generate_rules(basket):
        freq_items = apriori(basket, min_support=0.02, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.4)
        rules = rules[
            (rules['antecedents'].apply(len) == 1) &
            (rules['consequents'].apply(len) == 1) &
            (rules['support'] >= 0.02) &
            (rules['confidence'] >= 0.4) &
            (rules['lift'] >= 1.0)
        ]
        return rules

    def find_column(cols, keywords):
        for col in cols:
            for kw in keywords:
                norm_col = col.lower().replace("_", " ").replace("-", " ")
                norm_kw = kw.lower().replace("_", " ").replace("-", " ")
                if norm_kw in norm_col:
                    return col
        return None

    uploaded_file = st.file_uploader("📂 Upload your transaction CSV file", type=["csv"], key="market")
    if uploaded_file is None:
        st.info("📌 Please upload a CSV file with transaction data (Bill No + Item Name columns).")
        return

    with st.spinner("Loading and processing data..."):
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(uploaded_file)

    invoice_col = find_column(df.columns.tolist(), ["bill no", "billno", "invoice", "invoice no", "transaction id"])
    product_col = find_column(df.columns.tolist(), ["item name", "item", "product", "product name"])

    if invoice_col is None or product_col is None:
        st.error("❌ CSV must contain columns like 'Bill No' and 'Item Name'.")
        st.stop()

    item_counts = df[product_col].value_counts()
    common_items = item_counts[item_counts > 50].index
    df = df[df[product_col].isin(common_items)]

    progress_bar = st.progress(0)

    basket = df.groupby([invoice_col, product_col]).size().unstack(fill_value=0)
    progress_bar.progress(25)

    basket = basket > 0
    progress_bar.progress(50)

    rules = generate_rules(basket)
    progress_bar.progress(75)

    if rules.empty:
        st.warning("⚠️ No association rules found.")
        return

    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: next(iter(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: next(iter(x)))
    progress_bar.progress(100)

    st.subheader(f"📊 Product Pairs Bought Together ({len(rules)})")
    st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']])

    fig = px.scatter(
        rules,
        x='confidence',
        y='lift',
        size='support',
        color='support',
        hover_name='antecedents_str',
        hover_data=['consequents_str'],
        title="🧠 Lift vs Confidence for Product Pairs"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Product Pair Recommender")
    products = sorted(set(rules['antecedents_str']).union(rules['consequents_str']))
    selected_product = st.selectbox("Select a product to get recommended pairs", products)

    recommendations = rules[rules['antecedents_str'] == selected_product].sort_values(by='confidence', ascending=False)

    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            st.markdown(
                f"**If customer buys:** `{row['antecedents_str']}`  ➡️ "
                f"**Recommend:** `{row['consequents_str']}`  | "
                f"Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}"
            )
    else:
        st.info("No recommendations found for selected product.")

    csv = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Product Pairs CSV", csv, file_name='product_pairs.csv', mime='text/csv')


# Only call if running directly (not imported)
if __name__ == "__main__":
    try:
        run_marketbasket()
    except Exception as e:
        st.error(f"💥 Unexpected error: {e}")
