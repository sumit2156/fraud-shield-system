import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from agent import assess_transaction_and_investigate, model

st.set_page_config(page_title="Autonomous Fraud Forensics Copilot", layout="wide")

st.title("🛡️ Autonomous Fraud Investigation & Triage Copilot")
st.markdown("Combines **XGBoost Machine Learning** with an **Autonomous Investigation Agent** and **Human-in-the-Loop approval**.")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Inbound Transaction Details")
    user_id = st.text_input("Customer ID", value="USR_101")
    amount = st.number_input("Transaction Amount ($)", value=2850.0, step=50.0)
    dist = st.number_input("Distance from Registered Home (km)", value=1200.0, step=10.0)
    failed_logins = st.slider("Failed Password Attempts in last 24h", 0, 10, value=3)
    foreign_ip = st.selectbox("Foreign IP Detected?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    run_button = st.button("🚨 Process & Triage Transaction", type="primary")

# Prepare features DataFrame for model prediction and SHAP
features = pd.DataFrame([{
    'amount': amount,
    'distance_from_home_km': dist,
    'failed_logins_last_24h': failed_logins,
    'foreign_ip_flag': foreign_ip
}])

# Maintain session state for analyst actions and triage results
if "result" not in st.session_state:
    st.session_state.result = None
if "action_status" not in st.session_state:
    st.session_state.action_status = None

if run_button:
    with st.spinner("Analyzing risk & executing forensic tools..."):
        st.session_state.result = assess_transaction_and_investigate(user_id, amount, dist, failed_logins, foreign_ip)
        st.session_state.action_status = None

with col2:
    st.subheader("2. AI Agent Triage Results")
    
    if st.session_state.result:
        result = st.session_state.result
        score = result["risk_score"]
        st.metric(label="Fraud Risk Probability", value=f"{score*100:.1f}%")
        
        if result["status"] == "APPROVED":
            st.success("✅ " + result["message"])
        else:
            st.error("⚠️ ESCALATED: Suspicious Activity Detected!")
            st.code(result["report"], language="text")
            
            st.markdown("### 3. Human-in-the-Loop Analyst Action")
            c1, c2 = st.columns(2)
            if c1.button("⛔ Approve Block & Freeze Account"):
                st.session_state.action_status = f"Action Confirmed: Account {user_id} frozen. Webhook dispatched to security."
            if c2.button("✔️ Dismiss as False Positive"):
                st.session_state.action_status = "Action Dismissed: Recorded as False Positive for model retraining."
            
            if st.session_state.action_status:
                if "Confirmed" in st.session_state.action_status:
                    st.success(st.session_state.action_status)
                else:
                    st.warning(st.session_state.action_status)

st.divider()
st.subheader("🔍 Model Decision Attribution (SHAP)")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(features)
    fig, ax = plt.subplots(figsize=(8, 3))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()
except Exception as e:
    st.info(f"SHAP explanation: {e}")