import streamlit as st

st.set_page_config(page_title="Market analysis", layout="wide")

st.sidebar.title("Navigation")
app = st.sidebar.radio("Go to", ["Market Basket", "Customer Lifetime Value", "Sentiment Analysis"])

if app == "Market Basket":
    import marketbasketf
    marketbasketf.run_marketbasket()

elif app == "Customer Lifetime Value":
    import customerf
    customerf.run_clv()

elif app == "Sentiment Analysis":
    import sentimentf
    sentimentf.run_sentiment()
