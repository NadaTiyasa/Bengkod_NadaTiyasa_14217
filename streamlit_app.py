import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========================== CONFIG & THEME ==========================
st.set_page_config(page_title="Obesity Level Prediction", layout="centered")

def apply_blue_theme():
    st.markdown("""
        <style>
        .stApp {
            background-color: #e8f0fe;
        }
        h1 {
            color: #2a4d8f;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
        }
        h2, h3 {
            color: #3366cc;
            font-family: 'Segoe UI', sans-serif;
        }
        label, p, div, span {
            color: #222 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: #1a73e8;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #0f5edc;
            transition: 0.3s ease-in-out;
        }
        .stSlider > div > div {
            background-color: #1a73e8 !important;
        }
        .stAlert {
            background-color: #dbeafe;
            border-left: 6px solid #2563eb;
        }
        </style>
    """, unsafe_allow_html=True)

apply_blue_theme()

# ========================== LOAD MODEL ==========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("💡 Prediksi Tingkat Obesitas")
st.markdown("Masukkan data berikut untuk memprediksi tingkat obesitas Anda.")

# ========================== FORM INPUT ==========================
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

# ========================== PREDIKSI & OUTPUT ==========================
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
    user_input = user_input[[  # urutkan sesuai model
        'Age', 'Gender', 'Weight', 'CALC', 'FAVC', 'FCVC', 'SCC',
        'CH2O', 'family_history_with_overweight', 'FAF', 'CAEC'
    ]]

    X_scaled = scaler.transform(user_input)
    prediction = model.predict(X_scaled)
    result = label_encoder.inverse_transform(prediction)[0]
    kategori = result.replace("_", " ")

    # ========================== REKOMENDASI ==========================
    rekomendasi = {
        "Insufficient Weight": "🍽️ Perbanyak konsumsi kalori sehat, tambahkan protein dan lemak baik. 💪 Lakukan olahraga kekuatan untuk meningkatkan massa otot.",
        "Normal Weight": "✅ Pertahankan gaya hidup saat ini dengan pola makan seimbang dan aktivitas fisik rutin. Jangan lupakan hidrasi dan tidur cukup.",
        "Overweight Level I": "⚠️ Kurangi gula, makanan olahan, dan perbanyak aktivitas fisik seperti jalan kaki cepat atau bersepeda.",
        "Overweight Level II": "⚠️ Mulai atur pola makan dan tingkatkan frekuensi olahraga. Konsultasi gizi disarankan.",
        "Obesity Type I": "🚨 Terapkan diet rendah kalori, rutin olahraga, dan pertimbangkan konsultasi dengan ahli gizi atau dokter.",
        "Obesity Type II": "🚨 Perlu bimbingan profesional. Kurangi makanan tinggi kalori, perbanyak sayuran, dan lakukan aktivitas fisik ringan tapi konsisten.",
        "Obesity Type III": "🛑 Obesitas parah. Butuh intervensi medis. Ikuti program penurunan berat badan secara profesional dan terstruktur."
    }

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi:")
    st.success(f"Tingkat obesitas Anda diprediksi sebagai: **{kategori}**")

    st.markdown("### 💡 Rekomendasi Gaya Hidup")
    st.info(rekomendasi.get(kategori, "Tidak ada rekomendasi untuk kategori ini."))

    with st.expander("🔍 Lihat data yang dimasukkan"):
        st.dataframe(user_input)
