# %%
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("hourly_sheet.csv")

def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "")
    df = df.fillna("unknown")

    # Label encode categorical features
    label_cols = ['shift', 'machineworkstation_id', 'operator_name', 'product_name__part_number', 'reason_for_downtime']
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df

df_clean = preprocess(df)

# ---- Downtime Prediction Model ---- #
features = ['machineworkstation_id', 'operator_name', 'actual_output', 'target_output', 'defectsrework_units']
df_clean['downtime_flag'] = (df_clean['downtime_minutes'].astype(str).astype(float) > 5).astype(int)

X = df_clean[features]
y = df_clean['downtime_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

downtime_model = RandomForestClassifier()
downtime_model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/downtime_model.pkl", "wb") as f:
    pickle.dump(downtime_model, f)

# ---- Anomaly Detection Model ---- #
anomaly_features = ['actual_output', 'target_output', 'defectsrework_units', 'downtime_minutes']
df_clean[anomaly_features] = df_clean[anomaly_features].astype(str).astype(float)

anomaly_model = IsolationForest(contamination=0.1, random_state=42)
anomaly_model.fit(df_clean[anomaly_features])

with open("models/anomaly_model.pkl", "wb") as f:
    pickle.dump(anomaly_model, f)

print("✅ Models trained and saved.")
