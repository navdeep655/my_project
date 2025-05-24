import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import os

# Load tokenizer and label encoder
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load model
model = load_model("sentiment_model.h5")

MAX_LEN = 100
CSV_FILE = "user_reviews.csv"

# Create CSV with Date column if it doesn't exist
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Name", "Age", "City", "Country", "Review", "Sentiment", "Date"]).to_csv(CSV_FILE, index=False)

# --- Initialize session state to clear fields ---
for field in ["name", "age", "city", "country", "review"]:
    if field not in st.session_state:
        st.session_state[field] = "" if field != "age" else 1

def reset_form():
    for key in ["name", "age", "city", "country", "review"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()




# --- Streamlit UI ---
st.set_page_config(page_title="Review Collector", page_icon="🧠")
st.title("🧠 Share Your Review")
st.write("Please enter your details and share your review. Your feedback helps improve our service!")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("👤 Name", placeholder="Enter your name", key="name")
    age = st.number_input("🎂 Age", min_value=1, max_value=120, step=1, key="age")

with col2:
    city = st.text_input("🏙️ City", placeholder="Enter your city", key="city")
    country = st.text_input("🌍 Country", placeholder="Enter your country", key="country")

review = st.text_area("💬 Write your review here", placeholder="Type your review here...", key="review")

# --- Submit ---
if st.button("Submit"):
    if not all([name.strip(), str(age).strip(), city.strip(), country.strip(), review.strip()]):
        st.warning("⚠️ Please fill in all fields before submitting.")
    else:
        # Sentiment prediction
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
        prediction = model.predict(padded)
        sentiment = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")

        # Save data with date
        new_row = pd.DataFrame([[name, age, city, country, review, sentiment, date]],
                               columns=["Name", "Age", "City", "Country", "Review", "Sentiment", "Date"])
        new_row.to_csv(CSV_FILE, mode='a', header=False, index=False)

        st.success("✅ Thanks for your feedback!")
        

        # Clear the form inputs
        reset_form()
