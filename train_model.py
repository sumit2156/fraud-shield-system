import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, classification_report
import joblib

# 1. Create realistic synthetic banking data
np.random.seed(42)
n_samples = 10000

# Normal customer behavior
amounts = np.random.exponential(scale=50, size=n_samples)
dist_home = np.random.exponential(scale=10, size=n_samples)
failed_logins = np.random.poisson(lam=0.2, size=n_samples)
foreign_ip = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
is_fraud = np.zeros(n_samples, dtype=int)

# Inject 300 realistic fraud cases (high amount, strange IP, multiple failed logins)
fraud_indices = np.random.choice(n_samples, size=300, replace=False)
amounts[fraud_indices] = np.random.uniform(1000, 5000, size=300)
dist_home[fraud_indices] = np.random.uniform(500, 3000, size=300)
failed_logins[fraud_indices] = np.random.poisson(lam=3.5, size=300)
foreign_ip[fraud_indices] = np.random.choice([0, 1], size=300, p=[0.1, 0.9])
is_fraud[fraud_indices] = 1

df = pd.DataFrame({
    'amount': amounts,
    'distance_from_home_km': dist_home,
    'failed_logins_last_24h': failed_logins,
    'foreign_ip_flag': foreign_ip,
    'is_fraud': is_fraud
})

print(f"Dataset created: {len(df)} transactions, {sum(is_fraud)} fraudulent cases.")

# 2. Train/Test Split
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Train XGBoost Model
# scale_pos_weight handles the heavy imbalance (most transactions are legitimate)
scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    eval_metric='aucpr',
    random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate
y_probs = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, y_probs)
print(f"Model Training Complete! PR-AUC Score: {pr_auc:.4f}")

# 5. Save the trained model to disk
joblib.dump(model, "fraud_xgb_model.pkl")
df.to_csv("transactions_sample.csv", index=False)
print("Saved: 'fraud_xgb_model.pkl' and 'transactions_sample.csv'")