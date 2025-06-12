import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========================== CONFIG & THEME ==========================
st.set_page_config(page_title="Cek Tingkat Obesitas Anda", layout="centered")

def apply_custom_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #F5D5E0; }
        h1 { color: #210635; font-family: 'Segoe UI', sans-serif; text-align: center; }
        h2, h3 { color: #7B337E; font-family: 'Segoe UI', sans-serif; }
        label, p, div, span { color: #420D4B !important; font-family: 'Segoe UI', sans-serif; }
        .stButton > button {
            background-color: #6667AB;
            color: white; font-weight: bold;
            border-radius: 8px; border: none;
        }
        .stButton > button:hover {
            background-color: #7B337E;
            transition: 0.3s ease-in-out;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# ========================== LOAD MODEL ==========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ========================== SIDEBAR MENU ==========================
menu = st.sidebar.selectbox("📌 Menu", ["🔍 Prediksi Obesitas", "📂 Riwayat Prediksi"])

# ========================== SESSION STATE ==========================
if "riwayat_input" not in st.session_state:
    st.session_state.riwayat_input = []

# ========================== MENU 1: PREDIKSI ==========================
if menu == "🔍 Prediksi Obesitas":
    st.title("💡 Cek Tingkat Obesitas Anda")
    st.markdown("Masukkan data berikut untuk memprediksi tingkat obesitas berdasarkan gaya hidup Anda.")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Usia", 10, 100, 25)
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            height = st.number_input("Tinggi Badan (cm)", 100, 250, 170)
            weight = st.number_input("Berat Badan (kg)", 20, 200, 70)
            favc = st.selectbox("Sering Makan Tinggi Kalori?", ["Ya", "Tidak"])
            fcvc = st.number_input("Konsumsi Sayur (1–3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

        with col2:
            scc = st.selectbox("Pantau Kalori Harian?", ["Ya", "Tidak"])
            calc = st.selectbox("Konsumsi Alkohol", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])
            ch2o = st.number_input("Konsumsi Air (liter/hari)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
            fhwo = st.selectbox("Riwayat Keluarga Overweight?", ["Ya", "Tidak"])
            faf = st.number_input("Aktivitas Fisik Mingguan (jam)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
            caec = st.selectbox("Ngemil?", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])

        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        input_dict = {
            "Age": age,
            "Gender": 1 if gender == "Laki-laki" else 0,
            "Height": height / 100,  # konversi cm ke meter
            "Weight": weight,
            "CALC": {"Tidak": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}[calc],
            "FAVC": 1 if favc == "Ya" else 0,
            "FCVC": fcvc,
            "SCC": 1 if scc == "Ya" else 0,
            "CH2O": ch2o,
            "family_history_with_overweight": 1 if fhwo == "Ya" else 0,
            "FAF": faf,
            "CAEC": {"Tidak": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}[caec]
        }

        user_input = pd.DataFrame([input_dict])
        user_input = user_input[['Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'SCC',
                                 'CH2O', 'family_history_with_overweight', 'FAF', 'CAEC']]
        X_scaled = scaler.transform(user_input)
        prediction = model.predict(X_scaled)
        result = label_encoder.inverse_transform(prediction)[0]
        kategori = result.replace("_", " ")

        # Simpan ke riwayat
        input_dict["Kategori"] = kategori
        st.session_state.riwayat_input.append(input_dict)

        # Hasil
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi:")
        st.success(f"Tingkat obesitas Anda diprediksi sebagai: **{kategori}**")

        rekomendasi = {
            "Insufficient Weight": "🍽️ Perbanyak konsumsi kalori sehat, tambahkan protein dan lemak baik. 💪 Lakukan olahraga kekuatan untuk meningkatkan massa otot.",
            "Normal Weight": "✅ Pertahankan gaya hidup saat ini dengan pola makan seimbang dan aktivitas fisik rutin. Jangan lupakan hidrasi dan tidur cukup.",
            "Overweight Level I": "⚠️ Kurangi gula, makanan olahan, dan perbanyak aktivitas fisik seperti jalan kaki cepat atau bersepeda.",
            "Overweight Level II": "⚠️ Mulai atur pola makan dan tingkatkan frekuensi olahraga. Konsultasi gizi disarankan.",
            "Obesity Type I": "🚨 Terapkan diet rendah kalori, rutin olahraga, dan pertimbangkan konsultasi dengan ahli gizi atau dokter.",
            "Obesity Type II": "🚨 Perlu bimbingan profesional. Kurangi makanan tinggi kalori, perbanyak sayuran, dan lakukan aktivitas fisik ringan tapi konsisten.",
            "Obesity Type III": "🛑 Obesitas parah. Butuh intervensi medis. Ikuti program penurunan berat badan secara profesional dan terstruktur."
        }

        st.markdown("### 💡 Rekomendasi Gaya Hidup")
        st.info(rekomendasi.get(kategori, "Tidak ada rekomendasi untuk kategori ini."))

# ========================== MENU 2: RIWAYAT ==========================
if menu == "📂 Riwayat Prediksi":
    st.title("📂 Riwayat Input dan Kategori Obesitas")

    if st.session_state.riwayat_input:
        df_riwayat = pd.DataFrame(st.session_state.riwayat_input)
        st.dataframe(df_riwayat)

        # Tombol reset
        if st.button("🔄 Reset Riwayat"):
            st.session_state.riwayat_input = []
            st.success("Riwayat berhasil dihapus.")
    else:
        st.info("Belum ada riwayat prediksi yang disimpan.")
