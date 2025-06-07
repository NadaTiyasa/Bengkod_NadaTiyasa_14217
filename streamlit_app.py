import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= Load Model & Assets =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Obesity Level Prediction", layout="centered")
st.title("💡 Prediksi Tingkat Obesitas")
st.markdown("Masukkan data berikut untuk memprediksi tingkat obesitas Anda.")

# ================= Input Form Layout =================
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia", 10, 100, 25)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        weight = st.number_input("Berat Badan (kg)", 20, 200, 70)
        favc = st.selectbox("Sering Makan Tinggi Kalori?", ["yes", "no"])
        fcvc = st.slider("Konsumsi Sayur (1–3)", 1.0, 3.0, 2.0)
        scc = st.selectbox("Pantau Kalori Harian?", ["yes", "no"])

    with col2:
        calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = st.slider("Konsumsi Air (liter/hari)", 0.0, 3.0, 2.0)
        fhwo = st.selectbox("Riwayat Keluarga Overweight?", ["yes", "no"])
        faf = st.slider("Aktivitas Fisik Mingguan (jam)", 0.0, 3.0, 1.0)
        caec = st.selectbox("Ngemil?", ["no", "Sometimes", "Frequently", "Always"])

    submitted = st.form_submit_button("🔍 Prediksi")

# ================= Preprocessing & Prediction =================
if submitted:
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

    user_input = pd.DataFrame([input_dict])
    user_input = user_input[[  # pastikan urutan sesuai training
        'Age', 'Gender', 'Weight', 'CALC', 'FAVC', 'FCVC', 'SCC',
        'CH2O', 'family_history_with_overweight', 'FAF', 'CAEC'
    ]]

    X_scaled = scaler.transform(user_input)
    prediction = model.predict(X_scaled)
    result = label_encoder.inverse_transform(prediction)[0]

    # ================= Output =================
    st.markdown("---")
    st.subheader("📊 Hasil Prediksi:")
    st.success(f"Tingkat obesitas Anda diprediksi sebagai: **{result.replace('_', ' ')}**")

    # Optional: tampilkan dataframe input
    with st.expander("Lihat data yang dimasukkan"):
        st.dataframe(user_input)
