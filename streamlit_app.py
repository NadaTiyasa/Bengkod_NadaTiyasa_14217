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
            background-color: #fce4ec;
        }

        /* Ubah warna sidebar */
        section[data-testid="stSidebar"] {
            background-color: #94426a !important;
        }

        h1 {
            color: #620e2c;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
        }

        h2, h3 {
            color: #94426a;
            font-family: 'Segoe UI', sans-serif;
        }

        .stButton > button {
            background-color: #620e2c;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            margin: 5px;
            font-size: 16px;
            width: 100%;
            border: none;
        }

        .stButton > button:hover {
            background-color: #3f0a29;
            transition: 0.3s ease-in-out;
        }

        section[data-testid="stSidebar"] .stButton > button {
            background-color: #f5bcc1;
            color: #3f0a29;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: #3f0a29;
            color: white;
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

if "menu" not in st.session_state:
    st.session_state.menu = "prediksi"

# ========================== MENU PILIHAN ==========================
with st.sidebar:
    st.markdown("<h2 style='color:#620e2c;'>📌 Menu</h2>", unsafe_allow_html=True)
    if st.button("🔍 Prediksi Obesitas"):
        st.session_state.menu = "prediksi"
    if st.button("📂 Riwayat Prediksi"):
        st.session_state.menu = "riwayat"
    if st.button("📊 Statistik & Tren"):
        st.session_state.menu = "statistik"

# ========================== MENU 1: PREDIKSI ==========================
if st.session_state.menu == "prediksi":
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

        st.markdown(f"""
            <div style="background-color:#fce4ec; color:#620e2c; padding:20px; border-radius:15px; text-align:center; margin-top:20px; border: 2px solid #db90be;">
                <h3 style="margin-bottom:10px;">Tingkat obesitas Anda diprediksi sebagai:</h3>
                <h2 style="margin-top:0; color:#94426a;">{kategori}</h2>
            </div>
        """, unsafe_allow_html=True)
    
        rekomendasi = {
            "Insufficient Weight": "🍽️ Perbanyak konsumsi kalori sehat...",
            "Normal Weight": "✅ Pertahankan pola hidup sehat...",
            "Overweight Level I": "⚠️ Kurangi makanan tinggi gula dan lemak...",
            "Overweight Level II": "⚠️ Tambah frekuensi olahraga...",
            "Obesity Type I": "🚨 Konsultasi gizi dan program penurunan berat badan...",
            "Obesity Type II": "🚨 Intervensi profesional dibutuhkan...",
            "Obesity Type III": "🛑 Butuh penanganan medis intensif..."
        }
        
        st.markdown(f"""
            <div style="background-color:#fff0f5; color:#3f0a29; padding:15px; border-left: 5px solid #db90be; border-radius:10px; margin-top:15px;">
                {rekomendasi.get(kategori, "Tidak ada rekomendasi.")}
            </div>
        """, unsafe_allow_html=True)

# ========================== MENU 2: RIWAYAT ==========================
elif st.session_state.menu == "riwayat":
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
elif st.session_state.menu == "statistik":
    st.title("📊 Statistik & Tren dari Riwayat Prediksi")

    if st.session_state.riwayat_input:
        df = pd.DataFrame(st.session_state.riwayat_input)

        st.subheader("Distribusi Kategori Obesitas")
        fig1, ax1 = plt.subplots()
        df["Kategori"].value_counts().plot(kind='bar', color='#db90be', ax=ax1)
        ax1.set_ylabel("Jumlah")
        st.pyplot(fig1)

        st.subheader("Rata-rata Karakteristik Pengguna")
        st.dataframe(df[['Age', 'Height', 'Weight', 'FCVC', 'CH2O', 'FAF']].mean().round(2).rename("Rata-rata"))

        st.subheader("Distribusi Berat Badan per Kategori")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x="Kategori", y="Weight", ax=ax2, palette="pastel")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig2)

    else:
        st.warning("Belum ada data prediksi. Silakan lakukan prediksi terlebih dahulu.")

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #3f0a29;'>by <b>nadatiyasa</b> | NIM <b>A11.2022.14217</b></p>",
    unsafe_allow_html=True
)
