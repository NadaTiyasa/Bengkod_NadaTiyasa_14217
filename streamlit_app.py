import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model dan scaler
model = joblib.load("model_rf.pkl")  # Model terbaik kita adalah Random Forest
scaler = joblib.load("scaler.pkl")

# Judul
st.title("Prediksi Tingkat Obesitas")
st.markdown("Aplikasi ini memprediksi tingkat obesitas berdasarkan data fisik dan kebiasaan makan.")

# Input pengguna
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Usia", min_value=5, max_value=100, value=25)
height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=70.0)
family_history = st.selectbox("Riwayat keluarga dengan kelebihan berat badan", ["yes", "no"])
favc = st.selectbox("Sering makan tinggi kalori?", ["yes", "no"])
fcvc = st.slider("Konsumsi sayur tiap makan (1–3)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan besar per hari", 1.0, 4.0, 3.0)
caec = st.selectbox("Ngemil antar makan", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Apakah merokok?", ["yes", "no"])
ch2o = st.slider("Konsumsi air per hari (liter)", 0.0, 3.0, 2.0)
scc = st.selectbox("Memantau kalori?", ["yes", "no"])
faf = st.slider("Aktivitas fisik mingguan (jam)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu di depan layar (jam)", 0.0, 3.0, 2.0)
calc = st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Moda transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Mapping categorical to numeric sesuai training
def preprocess_input():
    input_dict = {
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": 1 if family_history == "yes" else 0,
        "FAVC": 1 if favc == "yes" else 0,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec],
        "SMOKE": 1 if smoke == "yes" else 0,
        "CH2O": ch2o,
        "SCC": 1 if scc == "yes" else 0,
        "FAF": faf,
        "TUE": tue,
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[calc],
        "MTRANS": {
            "Public_Transportation": 0,
            "Walking": 1,
            "Automobile": 2,
            "Motorbike": 3,
            "Bike": 4
        }[mtrans]
    }
    df_input = pd.DataFrame([input_dict])
    return df_input

if st.button("Prediksi"):
    user_input = preprocess_input()
    user_input_scaled = scaler.transform(user_input)
    pred = model.predict(user_input_scaled)
    label = joblib.load("label_encoder.pkl")
    st.success(f"Prediksi Tingkat Obesitas: **{label.inverse_transform(pred)[0]}**")
