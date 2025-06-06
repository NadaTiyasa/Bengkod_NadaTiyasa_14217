import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Prediksi Tingkat Obesitas")
st.markdown("Masukkan data di bawah untuk memprediksi tingkat obesitas.")

# Input pengguna
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Usia", 10, 100, 25)
height = st.number_input("Tinggi badan (m)", 1.0, 2.5, 1.7)
weight = st.number_input("Berat badan (kg)", 20, 200, 70)
favc = st.selectbox("Sering makan tinggi kalori?", ["yes", "no"])
fcvc = st.slider("Konsumsi sayur (1=jarang, 3=rutin)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan besar/hari", 1.0, 4.0, 3.0)
caec = st.selectbox("Ngemil?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
ch2o = st.slider("Konsumsi air per hari (liter)", 0.0, 3.0, 2.0)
faf = st.slider("Aktivitas fisik mingguan (jam)", 0.0, 3.0, 1.0)

# Preprocess input
def encode_input():
    input_dict = {
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "FAVC": 1 if favc == "yes" else 0,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec],
        "SMOKE": 1 if smoke == "yes" else 0,
        "CH2O": ch2o,
        "FAF": faf,
    }
    df = pd.DataFrame([input_dict])
    return df

if st.button("Prediksi"):
    user_input = encode_input()
    X_scaled = scaler.transform(user_input)
    prediction = model.predict(X_scaled)
    kelas = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Tingkat obesitas Anda: **{kelas}**")
