import pandas as pd
import pickle

def load_data():
    df = pd.read_csv("hourly_sheet.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "")
    return df.fillna("unknown")

def predict_downtime(machine_id, operator_name, actual_output, target_output, defects):
    with open("models/downtime_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load & preprocess data to get encodings
    df = load_data()

    # Label encode using current dataset
    from sklearn.preprocessing import LabelEncoder
    le_machine = LabelEncoder().fit(df['machineworkstation_id'])
    le_operator = LabelEncoder().fit(df['operator_name'])

    machine_encoded = le_machine.transform([machine_id])[0]
    operator_encoded = le_operator.transform([operator_name])[0]

    input_df = pd.DataFrame([[machine_encoded, operator_encoded, actual_output, target_output, defects]],
                            columns=['machineworkstation_id', 'operator_name', 'actual_output', 'target_output', 'defectsrework_units'])

    prob = model.predict_proba(input_df)[0][1]
    return round(prob * 100, 2)

def detect_anomalies():
    df = load_data()

    with open("models/anomaly_model.pkl", "rb") as f:
        model = pickle.load(f)

    df[["actual_output", "target_output", "defectsrework_units", "downtime_minutes"]] = df[["actual_output", "target_output", "defectsrework_units", "downtime_minutes"]].astype(str).astype(float)

    preds = model.predict(df[["actual_output", "target_output", "defectsrework_units", "downtime_minutes"]])
    df["anomaly"] = preds

    # Filter anomalies based on output drop > 25%
    anomaly_rows = df[df["anomaly"] == -1]
    messages = []
    for _, row in anomaly_rows.iterrows():
        drop_pct = round(100 - (row["actual_output"] / row["target_output"]) * 100, 2)
        
        # Show only anomalies with drop > 25%
        if drop_pct > 20:
            msg = f"⚠️ Machine {row['machineworkstation_id']} by operator {row['operator_name']} shows a {drop_pct}% drop in output. Possible issue detected!"
            messages.append(msg)

    return messages
