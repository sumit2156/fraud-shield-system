import os
import joblib
import pandas as pd
from database import query_user_security_logs

# Load the saved model
model = joblib.load("fraud_xgb_model.pkl")

def assess_transaction_and_investigate(user_id: str, amount: float, dist: float, failed_logins: int, foreign_ip: int):
    # 1. Prepare feature row
    features = pd.DataFrame([{
        'amount': amount,
        'distance_from_home_km': dist,
        'failed_logins_last_24h': failed_logins,
        'foreign_ip_flag': foreign_ip
    }])
    
    # 2. Get ML probability score
    fraud_prob = float(model.predict_proba(features)[0][1])
    
    # 3. Decision Boundary Check
    if fraud_prob < 0.60:
        return {
            "status": "APPROVED",
            "risk_score": fraud_prob,
            "message": "Transaction cleared normal risk bounds.",
            "features": features
        }
    
    # 4. If High Risk, Agent triggers forensic investigation tool
    print(f"[Agent] High risk detected ({fraud_prob:.2f}). Pulling security audit logs for {user_id}...")
    logs = query_user_security_logs(user_id)
    
    # 5. Agent drafts the Suspicious Activity Report (SAR)
    # Check if Groq API key is configured for LLM-powered SAR, else fallback to structured draft
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    ai_sar = None
    if groq_api_key and groq_api_key != "your_api_key_here":
        try:
            ai_sar = generate_ai_sar(user_id, amount, dist, logs, fraud_prob)
        except Exception as e:
            print(f"[Agent] Groq LLM failed: {e}")
            
    if ai_sar:
        sar_draft = ai_sar
    else:
        sar_draft = f"""======================================================
         SUSPICIOUS ACTIVITY REPORT (SAR)
======================================================
TARGET ACCOUNT     : {user_id}
MODEL RISK SCORE   : {fraud_prob * 100:.1f}% (CRITICAL)
FLAGGED ANOMALIES  :
  * Transaction Amount: ${amount:.2f} (Spike vs baseline)
  * Distance Anomaly  : {dist:.1f} km from registered home
  * Geo / IP Status   : Foreign IP Flag = {bool(foreign_ip)}

FORENSIC AUDIT TRAIL:
"""
        for action, ip, status in logs:
            sar_draft += f"  - [{status}] {action} from {ip}\n"
            
        sar_draft += """
AGENT RECOMMENDATION:
  * Suspected Account Takeover (ATO) followed by an unauthorized transaction.
  * Recommended Action: Halt payment, freeze digital tokens, require OTP re-authentication.
======================================================"""

    return {
        "status": "ESCALATED_FOR_REVIEW",
        "risk_score": fraud_prob,
        "report": sar_draft,
        "features": features
    }

def generate_ai_sar(user_id, amount, dist, logs, fraud_prob):
    api_key = os.getenv("GROQ_API_KEY", "your_api_key_here")
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=api_key)
    prompt = f"""
    You are an automated AML/Fraud compliance officer.
    A transaction for account {user_id} was flagged with risk score {fraud_prob*100:.1f}%.
    Transaction details: Amount ${amount}, Distance {dist} km from home.
    Forensic security logs: {logs}
    
    Write a formal 3-section Suspicious Activity Report (SAR):
    1. Incident Summary
    2. Chronological Attack Timeline
    3. Regulatory Recommendation
    Keep it concise and professional.
    """
    response = llm.invoke(prompt)
    return response.content
