import streamlit as st
import json
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# 🛠️ Page Config + Auto Refresh Time
st.set_page_config(page_title="Market Analysis", layout="wide")
st_autorefresh(interval=60000, key="refresh_timer")

# ⏱️ Show current time & date top-right
now = datetime.now()
current_time = now.strftime("%I:%M %p")
current_date = now.strftime("%d %B %Y")
st.markdown(f"<h5 style='text-align:right; color:gray;'>🕒 {current_time} | 📅 {current_date}</h5>", unsafe_allow_html=True)

# ---------------------- User File ----------------------
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

users = load_users()

# ---------------------- Session State ----------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "selected_app" not in st.session_state:
    st.session_state.selected_app = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"

# ---------------------- Authentication UI ----------------------
if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center; color: navy;'>Welcome to the Retail Analytics Application</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Please login to continue or create an account.</h4>", unsafe_allow_html=True)

    st.sidebar.title("Login / Signup")

    # 👉 Track mode change and clear inputs
    auth_mode = st.sidebar.selectbox("Choose", ["Login", "Sign Up", "Reset Password"])
    if auth_mode != st.session_state.auth_mode:
        st.session_state.auth_mode = auth_mode
        st.session_state.username_input = ""
        st.session_state.password_input = ""
        st.rerun()

    # 👉 Input fields with keys to track/reset
    username = st.sidebar.text_input("Username", key="username_input")
    password = st.sidebar.text_input("Password", type="password", key="password_input")

    if auth_mode == "Login":
        if st.sidebar.button("Login"):
            if username.strip() == "" or password.strip() == "":
                st.sidebar.warning("Please enter both username and password.")
            elif username in users and users[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")

    elif auth_mode == "Sign Up":
        if st.sidebar.button("Create Account"):
            if username.strip() == "" or password.strip() == "":
                st.sidebar.warning("Username and password cannot be empty.")
            elif username in users:
                st.sidebar.error("Username already exists.")
            else:
                users[username] = password
                save_users(users)
                st.sidebar.success("Account created! Please login.")

    elif auth_mode == "Reset Password":
        new_password = st.sidebar.text_input("New Password", type="password", key="reset_pwd")
        if st.sidebar.button("Reset Password"):
            if username.strip() == "" or new_password.strip() == "":
                st.sidebar.warning("Please enter both username and new password.")
            elif username in users:
                users[username] = new_password
                save_users(users)
                st.sidebar.success("Password reset successfully.")
            else:
                st.sidebar.error("Username not found.")

# ---------------------- AFTER LOGIN ----------------------
if st.session_state.authenticated:
    if st.session_state.selected_app is None:
        st.markdown("<h2 style='color: green;'>🎉 Welcome to Retail Analytics 🎉</h2>", unsafe_allow_html=True)
        st.info("This app includes:\n\n- 🛒 Market Basket\n- 💰 Customer Lifetime Value\n- 📊 CLV Dashboard\n- 😊 Sentiment Analysis")

    st.sidebar.markdown(f"### 👋 Hi, **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.selected_app = None
        st.rerun()

    st.sidebar.title("📂 Navigation")
    app = st.sidebar.radio("Go to", ["Market Basket", "CLV Dashboard", "Customer Lifetime Value", "Sentiment Analysis"], key="selected_app")

    if st.session_state.username == "admin":
        st.sidebar.markdown("🛡️ **Admin Access Enabled**")

    # 👉 Load modules
    if app == "Market Basket":
        import marketbasket; marketbasket.run_marketbasket()
    elif app == "CLV Dashboard":
        import clv; clv.run_clv_dashboard()
    elif app == "Customer Lifetime Value":
        import customer; customer.run_clv()
    elif app == "Sentiment Analysis":
        import sentiment; sentiment.run_sentiment()

    # Admin Panel
    # ---------------------- Admin Panel ----------------------
    if st.session_state.username == "admin":
        st.markdown("## 🔒 Admin Panel")

        st.write("### 👥 Registered Users")
        st.json(users)

        st.write("### 🧹 Delete a User")
        delete_user = st.selectbox("Select user to delete", [u for u in users if u != "admin"])
        if st.button("Delete User"):
            del users[delete_user]
            save_users(users)
            st.success(f"User '{delete_user}' deleted successfully.")
