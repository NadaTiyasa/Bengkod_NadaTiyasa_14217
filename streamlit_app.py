import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, dan label encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Prediksi Tingkat Obesitas")
st.markdown("Masukkan data di bawah untuk memprediksi tingkat obesitas.")

# Input pengguna (11 fitur sesuai model)
age = st.number_input("Usia", 10, 100, 25)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
weight = st.number_input("Berat Badan (kg)", 20, 200, 70)
calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
favc = st.selectbox("Sering Makan Tinggi Kalori?", ["yes", "no"])
fcvc = st.slider("Konsumsi Sayur (1–3)", 1.0, 3.0, 2.0)
scc = st.selectbox("Pantau Kalori Harian?", ["yes", "no"])
ch2o = st.slider("Konsumsi Air (liter per hari)", 0.0, 3.0, 2.0)
fhwo = st.selectbox("Riwayat Keluarga Overweight?", ["yes", "no"])
faf = st.slider("Aktivitas Fisik Mingguan (jam)", 0.0, 3.0, 1.0)
caec = st.selectbox("Ngemil?", ["no", "Sometimes", "Frequently", "Always"])

# Preprocessing input
def encode_input():
    input_dict = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Weight": weight,
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[calc],
        "FAVC": 1 if favc == "yes" else 0,
        "FCVC": fcvc,
        "SCC": 1 if scc == "yes" else 0,
        "CH2O": ch2o,
        "family_history_with_overweight": 1 if fhwo == "yes" else 0,
        "FAF": faf,
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]
    }
    df = pd.DataFrame([input_dict])
    return df

if st.button("Prediksi"):
    user_input = encode_input()
    # Sesuaikan urutan kolom agar cocok dengan saat pelatihan
    user_input = user_input[[
        'Age', 'Gender', 'Weight', 'CALC', 'FAVC', 'FCVC', 'SCC',
        'CH2O', 'family_history_with_overweight', 'FAF', 'CAEC'
    ]]
    X_scaled = scaler.transform(user_input)
    prediction = model.predict(X_scaled)
    result = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Tingkat obesitas Anda: **{result}**")
