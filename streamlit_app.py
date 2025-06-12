import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ========================== CONFIG & THEME ==========================
st.set_page_config(page_title="Obesity Predictor", layout="centered")

def apply_custom_theme():
    st.markdown("""
        <style>
        .stApp {
            background-color: #d4d994; /* Latar belakang umum */
        }

        h1 {
            color: #893941;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
        }

        h2, h3 {
            color: #5e6623;
            font-family: 'Segoe UI', sans-serif;
        }

        .stButton > button {
            background-color: #893941;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
        }

        .stButton > button:hover {
            background-color: #5e6623;
            transition: 0.3s ease-in-out;
        }

        .stRadio > div {
            background-color: #cb7885;
            padding: 10px;
            border-radius: 10px;
        }
        .stRadio > label {
            font-weight: bold;
            color: white;
            text-align: center;
            width: 100%;
            display: block;
        }

        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# ========================== LOAD MODEL ==========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ========================== SESSION STATE ==========================
if "riwayat_input" not in st.session_state:
    st.session_state.riwayat_input = []

# ========================== MENU PILIHAN ==========================
with st.sidebar:
    st.markdown("## 📌 Menu Utama")
    menu = st.radio("Silakan pilih:", ["🔍 Prediksi Obesitas", "📂 Riwayat Prediksi", "📊 Statistik & Tren"])

# ========================== MENU 1: PREDIKSI ==========================
if menu == "🔍 Prediksi Obesitas":
    st.title("💡 Prediksi Tingkat Obesitas Anda")
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Usia", 10, 100, 25)
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            height = st.number_input("Tinggi Badan (cm)", 100, 250, 170)
            weight = st.number_input("Berat Badan (kg)", 20, 200, 70)
            favc = st.selectbox("Sering Makan Tinggi Kalori?", ["Ya", "Tidak"])
            fcvc = st.number_input("Konsumsi Sayur (1–3)", 1.0, 3.0, 2.0, step=0.1)

        with col2:
            scc = st.selectbox("Pantau Kalori Harian?", ["Ya", "Tidak"])
            calc = st.selectbox("Konsumsi Alkohol", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])
            ch2o = st.number_input("Konsumsi Air (liter/hari)", 0.0, 5.0, 2.0, step=0.1)
            fhwo = st.selectbox("Riwayat Keluarga Overweight?", ["Ya", "Tidak"])
            faf = st.number_input("Aktivitas Fisik Mingguan (jam)", 0.0, 20.0, 1.0, step=0.5)
            caec = st.selectbox("Ngemil?", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])

        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        input_dict = {
            "Age": age,
            "Gender": 1 if gender == "Laki-laki" else 0,
            "Height": height / 100,
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
        X_scaled = scaler.transform(user_input)
        prediction = model.predict(X_scaled)
        result = label_encoder.inverse_transform(prediction)[0]
        kategori = result.replace("_", " ")

        input_dict["Kategori"] = kategori
        st.session_state.riwayat_input.append(input_dict)

        st.success(f"Tingkat obesitas Anda diprediksi sebagai: **{kategori}**")

        rekomendasi = {
            "Insufficient Weight": "🍽️ Perbanyak konsumsi kalori sehat...",
            "Normal Weight": "✅ Pertahankan pola hidup sehat...",
            "Overweight Level I": "⚠️ Kurangi makanan tinggi gula dan lemak...",
            "Overweight Level II": "⚠️ Tambah frekuensi olahraga...",
            "Obesity Type I": "🚨 Konsultasi gizi dan program penurunan berat badan...",
            "Obesity Type II": "🚨 Intervensi profesional dibutuhkan...",
            "Obesity Type III": "🛑 Butuh penanganan medis intensif..."
        }

        st.info(rekomendasi.get(kategori, "Tidak ada rekomendasi."))

# ========================== MENU 2: RIWAYAT ==========================
elif menu == "📂 Riwayat Prediksi":
    st.title("📂 Riwayat Prediksi Obesitas")

    if st.session_state.riwayat_input:
        df_riwayat = pd.DataFrame(st.session_state.riwayat_input)
        st.dataframe(df_riwayat)

        if st.button("🔄 Reset Riwayat"):
            st.session_state.riwayat_input = []
            st.success("Riwayat berhasil dihapus.")
    else:
        st.info("Belum ada riwayat prediksi yang disimpan.")

# ========================== MENU 3: STATISTIK ==========================
elif menu == "📊 Statistik & Tren":
    st.title("📊 Statistik & Tren dari Riwayat Prediksi")

    if st.session_state.riwayat_input:
        df = pd.DataFrame(st.session_state.riwayat_input)

        st.subheader("Distribusi Kategori Obesitas")
        fig1, ax1 = plt.subplots()
        df["Kategori"].value_counts().plot(kind='bar', color='#A74AC7', ax=ax1)
        ax1.set_ylabel("Jumlah")
        st.pyplot(fig1)

        st.subheader("Rata-rata Karakteristik Pengguna")
        st.dataframe(df[['Age', 'Height', 'Weight', 'FCVC', 'CH2O', 'FAF']].mean().round(2).rename("Rata-rata"))

        st.subheader("Distribusi Berat Badan per Kategori")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x="Kategori", y="Weight", ax=ax2, palette="magma")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig2)

    else:
        st.warning("Belum ada data prediksi. Silakan lakukan prediksi terlebih dahulu.")
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #5e6623;'>by <b>nadatiyasa</b> | NIM <b>A11.2022.14217</b></p>",
    unsafe_allow_html=True
)
