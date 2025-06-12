import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter
import matplotlib.pyplot as plt

# ========================== CONFIG & THEME ==========================
st.set_page_config(page_title="Cek Tingkat Obesitas Anda", layout="centered")

def apply_custom_theme():
    st.markdown("""
        <style>
        .stApp {
            background-color: #F5D5E0;
        }
        h1 {
            color: #210635;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
        }
        h2, h3 {
            color: #7B337E;
            font-family: 'Segoe UI', sans-serif;
        }
        label, p, div, span {
            color: #420D4B !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: #6667AB;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #7B337E;
            transition: 0.3s ease-in-out;
        }
        .stAlert {
            background-color: #f9ebf0;
            border-left: 6px solid #7B337E;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# ========================== LOAD MODEL ==========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("💡 Cek Tingkat Obesitas Anda")
st.markdown("Masukkan data berikut untuk memprediksi tingkat obesitas berdasarkan gaya hidup Anda.")

# ========================== FORM INPUT ==========================
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

# ========================== PREDIKSI & OUTPUT ==========================
if submitted:
    input_dict = {
        "Age": age,
        "Gender": 1 if gender == "Laki-laki" else 0,
        "Height": height,
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
    user_input = user_input[[  # urutkan sesuai model
        'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'SCC',
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

# ========================== VISUALISASI SIMULASI ==========================
st.markdown("## 📊 Distribusi Kategori Obesitas (Contoh Data)")

# Simulasi hasil prediksi dari beberapa data
hasil_prediksi = [
    "Normal_Weight", "Normal_Weight", "Normal_Weight",
    "Overweight_Level_I", "Normal_Weight"
]
pred_clean = [x.replace("_", " ") for x in hasil_prediksi]
count_pred = Counter(pred_clean)
df_vis = pd.DataFrame.from_dict(count_pred, orient='index', columns=["Jumlah"])
df_vis = df_vis.sort_values(by="Jumlah", ascending=False)

# Buat 2 kolom
col_vis, col_desc = st.columns([2, 1])  # Visualisasi 2x lebih lebar dari deskripsi

with col_vis:
    st.bar_chart(df_vis)

with col_desc:
    st.markdown("### ℹ️ Keterangan")
    st.markdown("""
    Grafik di samping menunjukkan **distribusi hasil prediksi** tingkat obesitas berdasarkan data simulasi.
    
    - Kategori **'Normal Weight'** mendominasi data simulasi.
    - Kategori lain seperti **'Overweight Level I'** juga muncul namun lebih sedikit.
    
    Visualisasi ini membantu memahami penyebaran kondisi gizi dari sekelompok data, dan dapat dikembangkan untuk data real pengguna secara agregat.
    """)

