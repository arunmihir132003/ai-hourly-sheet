import streamlit as st
from predict import predict_downtime, detect_anomalies
from datetime import datetime

st.title("🛠️ AI-Powered Hourly Sheet")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Feature", ["Downtime Prediction", "Anomaly Detection"])

# ----- DOWNTIME PREDICTION ----- #
if option == "Downtime Prediction":
    st.header("📉 Downtime Risk Prediction")
    st.markdown("Get predicted downtime risk based on machine and operator.")

    machine_id = st.selectbox("Select Machine ID", ["M101", "M102", "M103"])
    operator_name = st.selectbox("Select Operator", ["Meera", "Anita", "John", "Raj"])

    if st.button("Predict Downtime Risk"):
        risk = predict_downtime(machine_id, operator_name, actual_output=50, target_output=55, defects=2)  # fixed baseline
        st.success(f"🕒 Machine {machine_id} has a **{risk}%** risk of downtime in the next **2 hours** (as of {datetime.now().strftime('%H:%M')}).")

# ----- ANOMALY DETECTION ----- #
elif option == "Anomaly Detection":
    st.header("🚨 Anomaly Detection")
    st.markdown("Automatically detects performance drops using existing data.")

    if st.button("Detect Anomalies"):
        alerts = detect_anomalies()
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("✅ No anomalies detected!")
