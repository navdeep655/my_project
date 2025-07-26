import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import io

def run_sentiment():
    st.title("📈 Sentiment Visualizer")
    st.info("Please upload a CSV file containing columns similar to: Sentiment (or Text), Age, Time of Tweet, Country.")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key='sentiment')
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

            st.header("Map your dataset columns")
            cols = df.columns.tolist()

            # Detect sentiment column automatically (case insensitive)
            sentiment_col_candidates = ['Sentiment', 'sentiment', 'sentiments']
            sentiment_col = None
            for col in sentiment_col_candidates:
                if col in df.columns:
                    sentiment_col = col
                    break

            if sentiment_col is None:
                st.warning("No sentiment column found. Please select the text column to predict sentiment.")
                text_col = st.selectbox("Select Text Column for Sentiment Prediction", options=cols)
            else:
                st.success(f"Found Sentiment column: {sentiment_col}")
                text_col = None

            age_col = st.selectbox("Select Age column", options=cols)
            time_col = st.selectbox("Select Time of Tweet column", options=cols)
            country_col = st.selectbox("Select Country column", options=cols)

            if sentiment_col is None and not text_col:
                st.error("❌ Please select the text column for sentiment prediction.")
            elif not all([age_col, time_col, country_col]):
                st.error("❌ Please select all required columns.")
            else:
                # Prepare dataframe with renamed columns
                rename_map = {
                    age_col: 'Age',
                    time_col: 'Time of Tweet',
                    country_col: 'Country'
                }
                if sentiment_col:
                    rename_map[sentiment_col] = 'Sentiment'
                df_renamed = df.rename(columns=rename_map)

                # If no sentiment column, predict sentiment using TextBlob
                if sentiment_col is None:
                    def get_sentiment(text):
                        if pd.isna(text):
                            return "Neutral"
                        polarity = TextBlob(str(text)).sentiment.polarity
                        if polarity > 0.1:
                            return "Positive"
                        elif polarity < -0.1:
                            return "Negative"
                        else:
                            return "Neutral"

                    df_renamed['Sentiment'] = df_renamed[text_col].apply(get_sentiment)
                    st.success("✅ Sentiment predicted automatically using TextBlob!")

                # Drop rows with missing values in required columns
                df_clean = df_renamed[['Sentiment', 'Age', 'Time of Tweet', 'Country']].dropna()

                # Convert Age ranges to numeric midpoint
                def age_to_midpoint(age_range):
                    try:
                        parts = str(age_range).split('-')
                        start = int(parts[0])
                        end = int(parts[1])
                        return (start + end) // 2
                    except:
                        return None

                df_clean['Age_numeric'] = df_clean['Age'].apply(age_to_midpoint)
                df_clean = df_clean.dropna(subset=['Age_numeric'])

                # --- Add CSV download button if sentiment was predicted ---
                if sentiment_col is None:
                    csv_buffer = io.StringIO()
                    df_clean.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="📥 Download CSV with Predicted Sentiment",
                        data=csv_data,
                        file_name="sentiment_predicted.csv",
                        mime="text/csv"
                    )

                st.info("I have done the analysis, let me show you graphically.")

                # --- FILTERS ---
                st.sidebar.header("Filter your data")

                # Age slider filter
                min_age = int(df_clean['Age_numeric'].min())
                max_age = int(df_clean['Age_numeric'].max())
                age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

                # Time of Tweet filter
                try:
                    df_clean['Time of Tweet'] = pd.to_datetime(df_clean['Time of Tweet'])
                    min_date = df_clean['Time of Tweet'].min()
                    max_date = df_clean['Time of Tweet'].max()
                    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        df_filtered = df_clean[
                            (df_clean['Age_numeric'] >= age_range[0]) & (df_clean['Age_numeric'] <= age_range[1]) &
                            (df_clean['Time of Tweet'] >= pd.Timestamp(start_date)) & (df_clean['Time of Tweet'] <= pd.Timestamp(end_date))
                        ]
                    else:
                        df_filtered = df_clean[
                            (df_clean['Age_numeric'] >= age_range[0]) & (df_clean['Age_numeric'] <= age_range[1])
                        ]
                except Exception:
                    time_options = df_clean['Time of Tweet'].unique().tolist()
                    selected_times = st.sidebar.multiselect("Select Time of Tweet", options=time_options, default=time_options)
                    df_filtered = df_clean[
                        (df_clean['Age_numeric'] >= age_range[0]) & (df_clean['Age_numeric'] <= age_range[1]) &
                        (df_clean['Time of Tweet'].isin(selected_times))
                    ]

                # Country filter in sidebar (dropdown)
                country_options = df_clean['Country'].unique().tolist()
                selected_country = st.sidebar.selectbox("Select Country", options=["All"] + country_options)
                if selected_country != "All":
                    df_filtered = df_filtered[df_filtered['Country'] == selected_country]

                st.write(f"### Filtered data contains {df_filtered.shape[0]} records.")

                if df_filtered.empty:
                    st.warning("No data available for the selected filters. Please adjust filters.")
                else:
                    graph_option = st.radio(
                        "Choose a graph to display:",
                        ('Sentiment Distribution',
                         'Sentiment by Age Group',
                         'Sentiment by Time of Tweet',
                         'Top 10 Countries with Most Negative Tweets')
                    )

                    if graph_option == 'Sentiment Distribution':
                        st.subheader("📊 Sentiment Distribution")
                        fig1, ax1 = plt.subplots()
                        sns.countplot(data=df_filtered, x='Sentiment', hue='Sentiment', palette="Set2", ax=ax1, legend=False)
                        ax1.set_title("Sentiment Distribution")
                        st.pyplot(fig1)

                    elif graph_option == 'Sentiment by Age Group':
                        st.subheader("👥 Sentiment Count by Age Group")
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        sns.countplot(data=df_filtered, x='Sentiment', hue='Age_numeric', ax=ax2)
                        ax2.set_title("Sentiment Count by Age Group")
                        st.pyplot(fig2)

                    elif graph_option == 'Sentiment by Time of Tweet':
                        st.subheader("⏰ Sentiment Count by Time of Tweet")
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        sns.countplot(data=df_filtered, x='Sentiment', hue='Time of Tweet', ax=ax3)
                        ax3.set_title("Sentiment Count by Time of Tweet")
                        st.pyplot(fig3)

                    else:  # Top 10 Countries with Most Negative Tweets
                        st.subheader("🌍 Top 10 Countries with Most Negative Tweets")
                        negative_df = df_filtered[df_filtered['Sentiment'].str.lower() == 'negative']
                        top_countries = negative_df['Country'].value_counts().head(10)
                        fig4, ax4 = plt.subplots(figsize=(10, 6))
                        top_countries.plot(kind='bar', color='teal', ax=ax4)
                        ax4.set_title("Top 10 Countries with Most Negative Tweets")
                        ax4.set_xlabel("Country")
                        ax4.set_ylabel("Count")
                        st.pyplot(fig4)

        except Exception as e:
            st.error(f"❌ File reading failed: {e}")
