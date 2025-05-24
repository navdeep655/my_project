import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import ast
import plotly.express as px

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

# CSS styling for horizontal radio buttons and styling
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

@st.cache_data
def load_rules():
    def parse_frozenset_string(s):
        if not isinstance(s, str):
            return s
        try:
            if s.startswith("frozenset(") and s.endswith(")"):
                inner = s[len("frozenset("):-1]
                return list(ast.literal_eval(inner))
            else:
                return ast.literal_eval(s)
        except Exception:
            return []

    rules = pd.read_csv("output.csv")
    rules['antecedents'] = rules['antecedents'].apply(parse_frozenset_string)
    rules['consequents'] = rules['consequents'].apply(parse_frozenset_string)
    rules = rules.rename(columns={'confidence': 'success_rate', 'support': 'combination_frequency'})
    rules['success_rate'] = (rules['success_rate'] * 100).round(2)
    rules['combination_frequency'] = (rules['combination_frequency'] * 100).round(2)
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    return rules

def render_basic_graph_and_table():
    rules = load_rules()
    st.subheader("Top Association Rules (Products Bought Together)")
    st.dataframe(rules[['antecedents_str', 'consequents_str', 'combination_frequency', 'success_rate', 'lift']].head(10))

    st.markdown("""
    These rules show combinations of products that are frequently bought together.  
    For example, if `antecedents` = **Tea Mug** and `consequents` = **Cookies**,  
    then customers who buy **Tea Mug** often also buy **Cookies**.
    """)

    st.subheader("📈 Product Relationship Graph")
    top_rules = rules.head(10)
    G = nx.DiGraph()
    for _, row in top_rules.iterrows():
        ants = row['antecedents']
        cons = row['consequents']
        if not ants or not cons:
            continue
        for ant in ants:
            for con in cons:
                G.add_edge(ant, con, weight=row['lift'])
    fig, ax = plt.subplots(figsize=(15, 6))
    pos = nx.spring_layout(G, k=1)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue',
            font_size=10, font_weight='bold', edge_color='gray', ax=ax)
    st.pyplot(fig)

def render_product_pair_advisor(min_success, min_frequency):
    rules = load_rules()
    st.subheader("🧠 Product Pair Recommender")
    st.markdown("Use smart product pairing insights to boost cross-sell performance.")

    filtered = rules[
        (rules['success_rate'] >= min_success) &
        (rules['combination_frequency'] >= min_frequency)
    ].sort_values('success_rate', ascending=False)

    st.markdown(f"### 📦 Showing {len(filtered)} Product Recommendations")
    if not filtered.empty:
        for _, row in filtered.iterrows():
            st.markdown(f"""
                #### 🛒 Bought: {row['antecedents_str']}
                **➕ Recommend:** {row['consequents_str']}  
                🎯 **Success Rate:** {row['success_rate']}%  
                📊 **Frequency:** {row['combination_frequency']}%
            """)
    else:
        st.warning("No recommendations meet current filter criteria.")

    if not filtered.empty:
        st.markdown("### 🔍 Visual Explorer")
        fig = px.scatter(
            filtered,
            x='combination_frequency',
            y='success_rate',
            size='lift',
            color='lift',
            hover_name='antecedents_str',
            hover_data={'consequents_str': True},
            title="Lift vs Frequency vs Success Rate"
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

st.title("🛒 Market Basket Analysis")

if os.path.exists("output.csv"):

    selected_tab = st.radio("Select View", ["📊 Rules & Graph", "🧠 Product Recommender"], horizontal=True)

    # Show sidebar filters ONLY for Product Recommender
    if selected_tab == "🧠 Product Recommender":
        st.sidebar.header("🔧 Recommendation Filters")
        st.sidebar.markdown("")  # spacing
        min_success = st.sidebar.slider("🎯 Minimum Success Rate (%)", 1, 100, 40)
        min_frequency = st.sidebar.slider("📊 Minimum Frequency (%)", 0.0, 5.0, 0.5, step=0.1, format="%.2f%%")
    else:
        min_success = 1       # minimum values so filter works properly if hidden
        min_frequency = 0.0

    if selected_tab == "📊 Rules & Graph":
        render_basic_graph_and_table()
    else:
        render_product_pair_advisor(min_success, min_frequency)

else:
    st.warning("Top rules CSV not found. Please generate rules first.")
