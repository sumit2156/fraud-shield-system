from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3
import os
from datetime import datetime

# Resolve directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "history.db")
MODEL_FILE = os.path.join(BASE_DIR, "fraud_xgb_model.pkl")

app = FastAPI(title="Fraud Shield API")

# Allow frontend to communicate with this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize SQLite Database & Table
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scan_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        amount REAL,
        risk_score REAL,
        verdict TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# 2. Load the trained ML model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found. Run train_model.py first!")

# 3. Define the input format
class TransactionData(BaseModel):
    user_id: str
    amount: float
    distance_from_home_km: float
    failed_logins_last_24h: int
    foreign_ip_flag: int

# Serve index.html at root (http://127.0.0.1:8000)
@app.get("/")
def home():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Fraud Detection Brain is Active and Online!"}

# 4. Analyze and Save to Database
@app.post("/analyze")
def analyze(data: TransactionData):
    features = pd.DataFrame([{
        "amount": data.amount,
        "distance_from_home_km": data.distance_from_home_km,
        "failed_logins_last_24h": data.failed_logins_last_24h,
        "foreign_ip_flag": data.foreign_ip_flag
    }])
    
    risk_score = float(model.predict_proba(features)[0][1])
    risk_percentage = round(risk_score * 100, 1)
    
    if risk_score < 0.40:
        verdict = "SAFE"
        advice = "Transaction is within normal behavior."
    elif risk_score < 0.70:
        verdict = "SUSPICIOUS"
        advice = "Requires OTP or secondary verification."
    else:
        verdict = "CRITICAL FRAUD"
        advice = "Block card immediately and contact customer."

    # Save to SQLite
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO scan_history (user_id, amount, risk_score, verdict, created_at) VALUES (?, ?, ?, ?, ?)",
        (data.user_id, data.amount, risk_percentage, verdict, timestamp)
    )
    conn.commit()
    conn.close()
        
    return {
        "account": data.user_id,
        "fraud_risk_percentage": risk_percentage,
        "verdict": verdict,
        "recommended_action": advice
    }

# 5. Endpoint to Fetch the Last 5 Scans
@app.get("/history")
def get_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, amount, risk_score, verdict, created_at FROM scan_history ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    conn.close()
    
    history_list = []
    for r in rows:
        history_list.append({
            "user_id": r[0],
            "amount": r[1],
            "risk_score": r[2],
            "verdict": r[3],
            "created_at": r[4]
        })
    return history_list

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
