# ==============================================================================
# SISTEM NALARA (Nalar & Arah Karier) - SMKN 1 BINTAN TIMUR
# File: app.py
# Aplikasi Streamlit: Implementasi Two-Stage Hybrid (K-Means 2D & Random Forest)
# Deskripsi: Decision Support System Studi Lanjut SMK Berbasis Two-Stage Hybrid
#            (K-Means Clustering 2D & Random Forest Classifier)
# ==============================================================================

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="NALARA - Sistem Rekomendasi Studi Lanjut SMK",
    page_icon="🎓",
    page_title="NALARA — Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi CSS Khusus untuk Tampilan Modern, Minimalis, dan Bebas AI Slop
st.markdown("""
<style>
    /* Tipografi dan Spasi Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Judul dan Subjudul */
    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
        color: #0f172a;
        font-weight: 600;
    }
    
    /* Card Container Bersih */
    .nalara-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    
    .nalara-card-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.75rem;
    }
    
    .nalara-badge-linier {
        display: inline-block;
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .nalara-badge-rescue {
        display: inline-block;
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Tombol Bersih */
    div.stButton > button:first-child {
        background-color: #0f172a;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.55rem 1.25rem;
        transition: background-color 0.15s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #1e293b;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path File Data dan Model Resmi NALARA
DIR_MODEL = os.path.join(BASE_DIR, "pipeline_177_data_mining", "output_model")
DIR_DATA = os.path.join(BASE_DIR, "pipeline_177_data_mining")

# ==============================================================================
# FUNGSI MEMUAT MODEL & PARAMETER
# ==============================================================================
@st.cache_resource
def muat_sumber_daya():
    model_file = os.path.join(DIR_MODEL, "model_random_forest_177.joblib")
    data_file = os.path.join(DIR_DATA, "dataset_terklaster_177.csv")
    pred_file = os.path.join(DIR_DATA, "hasil_prediksi_rf_177.csv")

    rf_model = joblib.load(model_file)
    df_data = pd.read_csv(data_file)
    df_pred = pd.read_csv(pred_file) if os.path.exists(pred_file) else None

    # Menghitung statistik mean dan std populasi untuk scaling Z-Score
    fitur_numerik = [
        'Matematika', 'Bahasa_Indonesia', 'Bahasa_Inggris', 'Produktif', 'Rata_Rapor',
        'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional', 'Skor_Minat_Kejuruan'
    ]
    means = df_data[fitur_numerik].mean()
    stds = df_data[fitur_numerik].std()

    # Titik Centroid K-Means 2D (Skala Z-score)
    c0_scaled = np.array([-0.82320427, 0.15817904])  # Mismatch (Cluster 0)
    c1_scaled = np.array([0.77797326, -0.14948789])  # Sesuai (Cluster 1)

    return rf_model, df_data, df_pred, means, stds, c0_scaled, c1_scaled

# ==============================================================================
# SIDEBAR NAVIGASI
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=110)
    st.title("NALARA")
    st.markdown("**Nalar & Arah Karier**  \n*SMKN 1 Bintan Timur*")
    st.divider()
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 1.35rem; font-weight: 700; letter-spacing: -0.03em; color: #0f172a;">NALARA</div>
        <div style="font-size: 0.8rem; font-weight: 500; color: #64748b; margin-top: 2px;">Decision Support System</div>
        <div style="font-size: 0.75rem; color: #94a3b8;">SMKN 1 Bintan Timur</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin-bottom: 1.25rem;">
        <div style="font-size: 0.75rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Spesifikasi Model</div>
        <div style="font-size: 0.8rem; color: #334155; margin-top: 0.35rem;">
            • <b>Tahap 1:</b> K-Means Clustering 2D<br>
            • <b>Tahap 2:</b> Random Forest (100 Trees)<br>
            • <b>Fitur:</b> 11 Variabel Masukan
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("🎯 **Arsitektur:** Two-Stage Hybrid  \n- Tahap 1: K-Means 2D  \n- Tahap 2: Random Forest (100 Pohon)")
    
    st.divider()
    menu = st.radio(
        "Menu Navigasi:",
        "Navigasi:",
        options=[
            "🎯 Simulasi Konseling Siswa",
            "📊 Dashboard Analitik Model",
            "📑 Data Hasil Diagnosa",
            "ℹ️ Tentang Sistem NALARA"
            "Simulasi Konseling Siswa",
            "Evaluasi & Kinerja Model",
            "Data Hasil Diagnosa",
            "Dokumentasi Metodologi"
        ],
        index=0
    )
    
    st.divider()
    st.caption("Dikembangkan untuk Tugas Mata Kuliah Penambangan Data (Semester 5 - Teknik Informatika UMRAH).")
    st.markdown(
        "<div style='font-size: 0.75rem; color: #94a3b8; line-height: 1.4;'>"
        "Proyek Data Mining — Teknik Informatika<br>Universitas Maritim Raja Ali Haji"
        "</div>",
        unsafe_allow_html=True
    )

# Muat data model resmi
rf_model, df_data, df_pred, means, stds, c0_scaled, c1_scaled = muat_sumber_daya()

# ==============================================================================
# MENU 1: SIMULASI KONSELING SISWA (INFERENSI UTAMA)
# MENU 1: SIMULASI KONSELING SISWA
# ==============================================================================
if menu == "🎯 Simulasi Konseling Siswa":
    st.header("🎯 Simulasi Bimbingan & Rekomendasi Karier Siswa")
if menu == "Simulasi Konseling Siswa":
    st.title("Simulasi Konseling dan Rekomendasi Program Studi")
    st.markdown(
        "Masukkan nilai akademik rapor dan asesmen minat psikometri Holland RIASEC siswa "
        "untuk mendapatkan **Diagnosa Kesesuaian K-Means** dan **Rekomendasi Rumpun Program Studi PT Random Forest**."
        "Masukkan parameter nilai akademik rapor dan asesmen minat Holland RIASEC untuk menjalankan inferensi "
        "model hibrida: diagnosa kesesuaian kejuruan dan proyeksi program studi perguruan tinggi."
    )
    st.divider()
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Identitas Siswa
    col_bio1, col_bio2 = st.columns([2, 2])
    with col_bio1:
        nama_siswa = st.text_input("Nama Siswa / Pseudonim:", value="Ahmad Fauzan")
        nama_siswa = st.text_input("Nama Siswa / ID Subjek:", value="Ahmad Fauzan")
    with col_bio2:
        jurusan_pilihan = st.selectbox(
            "Jurusan SMK Saat Ini:",
            "Konsentrasi Keahlian SMK Saat Ini:",
            options=[
                "Rekayasa Perangkat Lunak (RPL)",
                "Akuntansi (AKT)",
                "Perhotelan (PH)",
                "Desain Komunikasi Visual (DKV)",
                "Tata Busana (TB)"
            ],
            index=0
        )

    st.subheader("1. Nilai Rapor 4 Mata Pelajaran Kunci (Skala 0 - 100)")
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Section 1: Nilai Rapor
    st.subheader("1. Nilai Akademik Rapor")
    st.caption("Nilai rata-rata semester pada 4 mata pelajaran kunci (skala 0 - 100). Ambang batas KKM = 80.0.")
    c_rap1, c_rap2, c_rap3, c_rap4 = st.columns(4)
    with c_rap1:
        val_mat = st.slider("Matematika", min_value=50.0, max_value=100.0, value=85.0, step=0.5)
        val_mat = st.slider("Matematika", 50.0, 100.0, 85.0, 0.5)
    with c_rap2:
        val_bi = st.slider("Bahasa Indonesia", min_value=50.0, max_value=100.0, value=84.0, step=0.5)
        val_bi = st.slider("Bahasa Indonesia", 50.0, 100.0, 84.0, 0.5)
    with c_rap3:
        val_en = st.slider("Bahasa Inggris", min_value=50.0, max_value=100.0, value=86.0, step=0.5)
        val_en = st.slider("Bahasa Inggris", 50.0, 100.0, 86.0, 0.5)
    with c_rap4:
        val_prod = st.slider("Produktif Kejuruan", min_value=50.0, max_value=100.0, value=88.0, step=0.5)
        val_prod = st.slider("Produktif Kejuruan", 50.0, 100.0, 88.0, 0.5)

    rata_rapor = (val_mat + val_bi + val_en + val_prod) / 4.0
    st.info(f"**Rata-Rata 4 Mapel Kunci Siswa:** `{rata_rapor:.2f}` (Batas KKM Akademik = 80.0)")
    st.markdown(
        f"<div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.9rem; margin-top: 0.5rem; font-size: 0.85rem; color: #334155;'>"
        f"<b>Rata-Rata Rapor 4 Mapel Kunci:</b> <code>{rata_rapor:.2f}</code>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.subheader("2. Asesmen Minat Holland RIASEC (Skor Kumulatif 7 butir, Skala 7 - 35)")
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Section 2: Minat RIASEC
    st.subheader("2. Profil Minat Holland RIASEC")
    st.caption("Skor kumulatif subskala minat riil siswa (skala 7 - 35).")
    c_r1, c_r2, c_r3 = st.columns(3)
    with c_r1:
        val_r = st.slider("Realistic (Teknikal / Alat Fisik)", 7, 35, 24)
        val_s = st.slider("Social (Pelayanan / Edukasi / Orang)", 7, 35, 22)
        val_r = st.slider("Realistic (Keteknikan & Fisik)", 7, 35, 24)
        val_s = st.slider("Social (Sosial & Pelayanan)", 7, 35, 22)
    with c_r2:
        val_i = st.slider("Investigative (Analisis / Logika)", 7, 35, 31)
        val_e = st.slider("Enterprising (Bisnis / Manajerial)", 7, 35, 23)
        val_i = st.slider("Investigative (Analisis & Logika)", 7, 35, 31)
        val_e = st.slider("Enterprising (Bisnis & Kepemimpinan)", 7, 35, 23)
    with c_r3:
        val_a = st.slider("Artistic (Seni / Kreativitas / Desain)", 7, 35, 20)
        val_c = st.slider("Conventional (Administrasi / Terstruktur)", 7, 35, 25)
        val_a = st.slider("Artistic (Seni & Kreativitas)", 7, 35, 20)
        val_c = st.slider("Conventional (Administrasi & Keteraturan)", 7, 35, 25)

    # Identifikasi Skor Minat Kejuruan berdasarkan Jurusan
    jur_u = jurusan_pilihan.upper()
    if 'RPL' in jur_u:
        minat_kejuruan = val_i
        dim_nama = 'Investigative (RPL)'
        dim_nama = 'Investigative'
    elif 'AKUNTANSI' in jur_u or 'AKT' in jur_u:
        minat_kejuruan = val_c
        dim_nama = 'Conventional (Akuntansi)'
        dim_nama = 'Conventional'
    elif 'PERHOTELAN' in jur_u or 'PH' in jur_u:
        minat_kejuruan = val_s
        dim_nama = 'Social (Perhotelan)'
        dim_nama = 'Social'
    elif 'DKV' in jur_u or 'DESAIN' in jur_u:
        minat_kejuruan = val_a
        dim_nama = 'Artistic (DKV)'
        dim_nama = 'Artistic'
    elif 'BUSANA' in jur_u or 'TB' in jur_u:
        minat_kejuruan = val_a
        dim_nama = 'Artistic (Tata Busana)'
        dim_nama = 'Artistic'
    else:
        minat_kejuruan = val_r
        dim_nama = 'Realistic'

    btn_proses = st.button("🚀 Jalankan Analisis & Rekomendasi NALARA", type="primary", use_container_width=True)
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    btn_proses = st.button("Jalankan Analisis Rekomendasi", use_container_width=True)

    if btn_proses:
        st.divider()
        # Normalisasi Z-Score
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        # Normalisasi Z-Score Sumbu 2D K-Means
        x_scaled = (minat_kejuruan - means['Skor_Minat_Kejuruan']) / stds['Skor_Minat_Kejuruan']
        y_scaled = (rata_rapor - means['Rata_Rapor']) / stds['Rata_Rapor']

        # Stage 1: Diagnosa K-Means 2D
        d0 = np.sqrt((x_scaled - c0_scaled[0])**2 + (y_scaled - c0_scaled[1])**2)
        d1 = np.sqrt((x_scaled - c1_scaled[0])**2 + (y_scaled - c1_scaled[1])**2)
        klaster_kmeans = 1 if d1 < d0 else 0

        # Normalisasi 11 Fitur untuk Random Forest
        z_mat = (val_mat - means['Matematika']) / stds['Matematika']
        z_bi = (val_bi - means['Bahasa_Indonesia']) / stds['Bahasa_Indonesia']
        z_en = (val_en - means['Bahasa_Inggris']) / stds['Bahasa_Inggris']
        z_prod = (val_prod - means['Produktif']) / stds['Produktif']
        z_r = (val_r - means['Realistic']) / stds['Realistic']
        z_i = (val_i - means['Investigative']) / stds['Investigative']
        z_a = (val_a - means['Artistic']) / stds['Artistic']
        z_s = (val_s - means['Social']) / stds['Social']
        z_e = (val_e - means['Enterprising']) / stds['Enterprising']
        z_c = (val_c - means['Conventional']) / stds['Conventional']

        X_input = pd.DataFrame([{
            'Matematika_scaled': z_mat,
            'Bahasa_Indonesia_scaled': z_bi,
            'Bahasa_Inggris_scaled': z_en,
            'Produktif_scaled': z_prod,
            'Realistic_scaled': z_r,
            'Investigative_scaled': z_i,
            'Artistic_scaled': z_a,
            'Social_scaled': z_s,
            'Enterprising_scaled': z_e,
            'Conventional_scaled': z_c,
            'Klaster_KMeans': klaster_kmeans
        }])

        # Stage 2: Random Forest Prediction
        pred_prodi = rf_model.predict(X_input)[0]
        proba = rf_model.predict_proba(X_input)[0]
        kelas_rf = rf_model.classes_
        top_conf = np.max(proba) * 100

        # Tampilkan Hasil dalam Layout 2 Kolom
        # Tampilan Hasil Evaluasi dalam 2 Kolom Bersih
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("### 🔍 TAHAP 1: DIAGNOSA KESESUAIAN (K-MEANS 2D)")
            st.markdown("""
            <div class="nalara-card">
                <div class="nalara-card-header">Tahap 1: Diagnosa Kongruensi (K-Means 2D)</div>
            """, unsafe_allow_html=True)
            
            if klaster_kmeans == 1:
                st.success("✅ **STATUS: SESUAI JURUSAN (High Congruence)**")
                st.markdown("<span class='nalara-badge-linier'>Kongruen (Sesuai Kejuruan)</span>", unsafe_allow_html=True)
                st.markdown(
                    f"- **Skor Minat Kejuruan:** `{minat_kejuruan}` ({dim_nama})  \n"
                    f"- **Rata-Rata Rapor:** `{rata_rapor:.2f}`  \n"
                    f"- Jarak ke Centroid Sesuai ($C_1$): `{d1:.4f}` *(Lebih Dekat)*  \n"
                    f"- Jarak ke Centroid Mismatch ($C_0$): `{d0:.4f}`"
                    f"<div style='font-size: 0.85rem; color: #334155; margin-top: 0.8rem; line-height: 1.6;'>"
                    f"• <b>Skor Minat Kejuruan:</b> {minat_kejuruan} (Dimensi {dim_nama})<br>"
                    f"• <b>Rata-Rata Rapor:</b> {rata_rapor:.2f}<br>"
                    f"• <b>Jarak Centroid Sesuai (C1):</b> {d1:.4f} <i>(Posisi Lebih Dekat)</i><br>"
                    f"• <b>Jarak Centroid Mismatch (C0):</b> {d0:.4f}"
                    f"</div>"
                    f"<div style='font-size: 0.8rem; color: #64748b; margin-top: 0.75rem; border-top: 1px solid #f1f5f9; padding-top: 0.5rem;'>"
                    f"Profil minat kejuruan siswa sejalan dengan jurusan SMK yang ditempuh saat ini."
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.caption("💡 **Catatan BK:** Minat kejuruan siswa selaras dengan jurusan SMK saat ini.")
            else:
                st.warning("⚠️ **STATUS: TIDAK SESUAI / MISMATCH (Butuh Penyelamatan)**")
                st.markdown("<span class='nalara-badge-rescue'>Diskongruen (Perlu Rekomendasi Penyelamatan)</span>", unsafe_allow_html=True)
                st.markdown(
                    f"- **Skor Minat Kejuruan:** `{minat_kejuruan}` ({dim_nama})  \n"
                    f"- **Rata-Rata Rapor:** `{rata_rapor:.2f}`  \n"
                    f"- Jarak ke Centroid Mismatch ($C_0$): `{d0:.4f}` *(Lebih Dekat)*  \n"
                    f"- Jarak ke Centroid Sesuai ($C_1$): `{d1:.4f}`"
                    f"<div style='font-size: 0.85rem; color: #334155; margin-top: 0.8rem; line-height: 1.6;'>"
                    f"• <b>Skor Minat Kejuruan:</b> {minat_kejuruan} (Dimensi {dim_nama})<br>"
                    f"• <b>Rata-Rata Rapor:</b> {rata_rapor:.2f}<br>"
                    f"• <b>Jarak Centroid Mismatch (C0):</b> {d0:.4f} <i>(Posisi Lebih Dekat)</i><br>"
                    f"• <b>Jarak Centroid Sesuai (C1):</b> {d1:.4f}"
                    f"</div>"
                    f"<div style='font-size: 0.8rem; color: #64748b; margin-top: 0.75rem; border-top: 1px solid #f1f5f9; padding-top: 0.5rem;'>"
                    f"Teridentifikasi diskongruensi antara minat dan jurusan SMK; model memprioritaskan rekomendasi lintas minat."
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.caption("💡 **Catatan BK:** Siswa mengalami kesenjangan minat; butuh opsi lintas jurusan.")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("### 🎓 TAHAP 2: REKOMENDASI RUMPUN PRODI PT")
            st.markdown(f"#### 👉 **{pred_prodi}**")
            st.markdown("""
            <div class="nalara-card">
                <div class="nalara-card-header">Tahap 2: Rekomendasi Rumpun Program Studi PT</div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div style='font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem;'>{pred_prodi}</div>", unsafe_allow_html=True)
            
            if klaster_kmeans == 1:
                st.markdown("📘 **Tipe:** `Rekomendasi Linier Sejalur SMK`")
                st.markdown("<div style='font-size: 0.85rem; color: #166534; font-weight: 500; margin-bottom: 0.75rem;'>Kategori: Rekomendasi Linier Sejalur</div>", unsafe_allow_html=True)
            else:
                st.markdown("📙 **Tipe:** `Rekomendasi Penyelamatan (Lintas Minat Sejati)`")
                st.markdown("<div style='font-size: 0.85rem; color: #92400e; font-weight: 500; margin-bottom: 0.75rem;'>Kategori: Rekomendasi Penyelamatan (Lintas Minat Psikometri)</div>", unsafe_allow_html=True)

            st.metric("Tingkat Keyakinan Model (Confidence)", f"{top_conf:.2f}%")
            st.markdown(
                f"<div style='font-size: 0.85rem; color: #475569; margin-bottom: 0.35rem;'><b>Tingkat Keyakinan Model:</b> {top_conf:.2f}%</div>",
                unsafe_allow_html=True
            )
            st.progress(float(top_conf) / 100.0)
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 Distribusi Probabilitas 5 Rumpun Program Studi PT")
        # Bagian Grafik Distribusi Probabilitas
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.subheader("Distribusi Probabilitas Model")
        
        df_prob = pd.DataFrame({
            'Rumpun Program Studi': kelas_rf,
            'Probabilitas (%)': proba * 100
        }).sort_values('Probabilitas (%)', ascending=True)

        fig_p, ax_p = plt.subplots(figsize=(8, 3.5), dpi=200)
        bars = ax_p.barh(df_prob['Rumpun Program Studi'], df_prob['Probabilitas (%)'], color='#3498db', edgecolor='black')
        fig_p, ax_p = plt.subplots(figsize=(8, 3.2), dpi=200)
        fig_p.patch.set_facecolor('#ffffff')
        ax_p.set_facecolor('#ffffff')
        
        bars = ax_p.barh(
            df_prob['Rumpun Program Studi'], 
            df_prob['Probabilitas (%)'], 
            color='#334155', 
            height=0.6,
            edgecolor='none'
        )
        for bar in bars:
            w = bar.get_width()
            ax_p.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9, fontweight='bold')
            ax_p.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=8.5, color='#0f172a', fontweight='600')
            
        ax_p.set_xlim([0, 115])
        ax_p.set_xlabel("Tingkat Probabilitas Suara Pohon Keputusan (%)", fontsize=9, fontweight='bold')
        ax_p.set_xlabel("Tingkat Keyakinan Suara Pohon Keputusan (%)", fontsize=8.5, color='#475569', labelpad=8)
        ax_p.tick_params(axis='both', which='both', labelsize=8.5, colors='#334155')
        ax_p.spines['top'].set_visible(False)
        ax_p.spines['right'].set_visible(False)
        ax_p.spines['left'].set_color('#cbd5e1')
        ax_p.spines['bottom'].set_color('#cbd5e1')
        ax_p.grid(axis='x', linestyle='--', alpha=0.5, color='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig_p)

        # Rekapitulasi Suara Riil 100 Pohon untuk Siswa yang Baru Saja Disimulasikan
        st.divider()
        st.subheader("🗳️ Rekapitulasi Suara 100 Pohon Keputusan (Khusus Siswa Ini)")
        st.markdown(
            f"Berikut adalah hasil pemungutan suara (*majority voting*) langsung dari **seluruh 100 pohon keputusan** "
            f"terhadap profil siswa **{nama_siswa}** berdasarkan nilai rapor dan skor minat yang baru saja Anda atur:"
        )
        # Rekapitulasi Suara 100 Pohon Riil Khusus Siswa Ini
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.subheader("Rekapitulasi Suara 100 Pohon Keputusan (Simulasi Ini)")
        st.caption(f"Distribusi suara riil dari 100 estimator Random Forest untuk profil siswa: {nama_siswa}.")

        # Ekstraksi suara dari masing-masing 100 estimator pohon
        tree_preds = [estimator.predict(X_input.values)[0] for estimator in rf_model.estimators_]
        classes_list = list(rf_model.classes_)
        vote_dict = {cls_name: 0 for cls_name in classes_list}
        for v in tree_preds:
            cls_name = classes_list[int(v)]
            vote_dict[cls_name] += 1

        df_sim_votes = pd.DataFrame(list(vote_dict.items()), columns=['Rumpun Program Studi PT', 'Jumlah Suara Pohon (dari 100)'])
        df_sim_votes['Persentase Kesepakatan'] = df_sim_votes['Jumlah Suara Pohon (dari 100)'].apply(lambda x: f"{x}%")
        df_sim_votes['Konsensus (%)'] = df_sim_votes['Jumlah Suara Pohon (dari 100)'].apply(lambda x: f"{x}%")
        df_sim_votes = df_sim_votes.sort_values(by='Jumlah Suara Pohon (dari 100)', ascending=False).reset_index(drop=True)

        st.dataframe(df_sim_votes, use_container_width=True)
        st.info(
            f"💡 **Kesimpulan Konsensus Ensemble:** Dari total 100 pohon keputusan independen di dalam model Random Forest, "
            f"sebanyak **{vote_dict[pred_prodi]} pohon** ({vote_dict[pred_prodi]}%) sepakat memilih **{pred_prodi}** "
            f"sebagai rekomendasi paling ideal untuk siswa ini."
        st.dataframe(df_sim_votes, use_container_width=True, hide_index=True)
        st.markdown(
            f"<div style='font-size: 0.85rem; color: #475569; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem;'>"
            f"<b>Konsensus Ensemble:</b> Dari total 100 pohon keputusan, sebanyak <b>{vote_dict[pred_prodi]} pohon ({vote_dict[pred_prodi]}%)</b> sepakat memproyeksikan <b>{pred_prodi}</b>."
            f"</div>",
            unsafe_allow_html=True
        )

# ==============================================================================
# MENU 2: DASHBOARD ANALITIK MODEL
# MENU 2: EVALUASI & KINERJA MODEL
# ==============================================================================
elif menu == "📊 Dashboard Analitik Model":
    st.header("📊 Dashboard Analitik & Kinerja Model NALARA")
    st.markdown("Visualisasi grafis hasil pemodelan dan evaluasi empiris pada data siswa SMKN 1 Bintan Timur.")
    st.divider()
elif menu == "Evaluasi & Kinerja Model":
    st.title("Evaluasi Kinerja Model")
    st.markdown("Dokumentasi empiris hasil pelatihan dan evaluasi model Machine Learning pada dataset SMKN 1 Bintan Timur.")
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Metric Header
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Siswa Dianalisis", f"{len(df_data)} Siswa")
        st.metric("Total Sampel Siswa", f"{len(df_data)} Data")
    with m_col2:
        st.metric("Jumlah Jurusan SMK", "5 Jurusan")
        st.metric("Konsentrasi Keahlian", "5 Jurusan")
    with m_col3:
        st.metric("Silhouette Score K-Means", "0.3213")
        st.metric("Silhouette Score (K=2)", "0.3213")
    with m_col4:
        st.metric("Akurasi Random Forest", "86.11%")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Diagnosa K-Means 2D", 
        "🎯 Confusion Matrix & Akurasi", 
        "⭐ Feature Importance",
        "🌳 Pohon Keputusan (Decision Tree)"
        "Diagnosa K-Means 2D", 
        "Confusion Matrix", 
        "Feature Importance",
        "Arsitektur & Voting Pohon"
    ])

    with tab1:
        st.subheader("Visualisasi Klaster Kesesuaian Siswa (K-Means 2D)")
        img_scatter = os.path.join(DIR_MODEL, "scatter_kmeans_2d_177.png")
        img_sil = os.path.join(DIR_MODEL, "evaluasi_silhouette_kmeans_177.png")
        
        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            if os.path.exists(img_scatter):
                st.image(img_scatter, caption="Scatter Plot 2D (Minat vs Rapor) & Titik Centroid", use_container_width=True)
                st.image(img_scatter, caption="Scatter Plot 2D: Minat Kejuruan vs Rata-Rata Rapor", use_container_width=True)
        with c_sc2:
            if os.path.exists(img_sil):
                st.image(img_sil, caption="Silhouette Plot Pemisahan Klaster K=2", use_container_width=True)

    with tab2:
        st.subheader("Evaluasi Klasifikasi Random Forest (Confusion Matrix)")
        img_cm = os.path.join(DIR_MODEL, "confusion_matrix_rf_177.png")
        if os.path.exists(img_cm):
            st.image(img_cm, caption="Confusion Matrix Prediksi 5 Rumpun Program Studi PT", width=650)

    with tab3:
        st.subheader("Peringkat Fitur Paling Berpengaruh (Feature Importance)")
        st.subheader("Peringkat Kepentingan Fitur (Feature Importance)")
        img_fi = os.path.join(DIR_MODEL, "feature_importance_rf_177.png")
        if os.path.exists(img_fi):
            st.image(img_fi, caption="Gini Feature Importance (11 Variabel Masukan)", width=750)
            st.image(img_fi, caption="Gini Feature Importance pada 11 Variabel Masukan", width=750)

    with tab4:
        st.subheader("🌳 Diagram Struktur Pohon Keputusan (Random Forest)")
        st.subheader("Diagram Struktur Pohon & Mekanisme Voting")
        st.markdown(
            "Berikut adalah representasi visual diagram pohon keputusan model **Random Forest Classifier NALARA**: "
            "menggambarkan bagaimana 11 fitur masukan diproses melalui 100 pohon keputusan, mekanisme agregasi suara (*majority voting*), "
            "hingga menghasilkan rekomendasi akhir rumpun program studi."
            "Representasi arsitektur ensemble Random Forest Classifier NALARA: "
            "menggambarkan pemrosesan 11 fitur masukan melalui pohon keputusan dan agregasi suara mayoritas."
        )

        img_final_kotak = os.path.join(DIR_MODEL, "diagram_pohon_final_rf.jpg")
        img_final_lingkaran = os.path.join(DIR_MODEL, "diagram_pohon_lingkaran_rf.jpg")
        img_tree_asli = os.path.join(DIR_MODEL, "struktur_pohon_keputusan_asli_rf_177.png")

        pilihan_pohon = st.radio(
            "Pilih Tampilan Diagram:",
            "Pilih Format Visualisasi Diagram:",
            options=[
                "🌟 Diagram Arsitektur Final (Flowchart Kotak - Rekomendasi)",
                "🟠 Diagram Arsitektur Alternatif (Lingkaran / Path Tracing)",
                "🔬 Detail Percabangan Pohon #1 (Scikit-Learn Tree)"
                "Diagram Arsitektur (Flowchart Kotak — Rekomendasi Utama)",
                "Diagram Arsitektur (Graf Lingkaran & Path Tracing)",
                "Struktur Percabangan Pohon #1 (Scikit-Learn Tree)"
            ],
            index=0,
            horizontal=True
        )

        if "🌟" in pilihan_pohon:
        if "Flowchart" in pilihan_pohon:
            if os.path.exists(img_final_kotak):
                st.image(img_final_kotak, caption="Diagram Arsitektur & Mekanisme Kerja Random Forest (Versi Final Kotak)", use_container_width=True)
                st.image(img_final_kotak, caption="Diagram Arsitektur & Mekanisme Kerja Random Forest (Versi Flowchart)", use_container_width=True)
                with open(img_final_kotak, "rb") as f_img:
                    st.download_button("📥 Unduh Diagram Arsitektur Final (Kotak)", data=f_img, file_name="diagram_arsitektur_rf_final.jpg", mime="image/jpeg")
        elif "🟠" in pilihan_pohon:
                    st.download_button("Unduh Diagram Arsitektur (JPG)", data=f_img, file_name="diagram_arsitektur_rf_final.jpg", mime="image/jpeg")
        elif "Graf" in pilihan_pohon:
            if os.path.exists(img_final_lingkaran):
                st.image(img_final_lingkaran, caption="Diagram Arsitektur Alternatif (Lingkaran & Tracing Jalur Siswa_018)", use_container_width=True)
                st.image(img_final_lingkaran, caption="Diagram Arsitektur Alternatif (Graf Lingkaran & Penelusuran Jalur)", use_container_width=True)
                with open(img_final_lingkaran, "rb") as f_img:
                    st.download_button("📥 Unduh Diagram Alternatif (Lingkaran)", data=f_img, file_name="diagram_arsitektur_rf_lingkaran.jpg", mime="image/jpeg")
                    st.download_button("Unduh Diagram Alternatif (JPG)", data=f_img, file_name="diagram_arsitektur_rf_lingkaran.jpg", mime="image/jpeg")
        else:
            if os.path.exists(img_tree_asli):
                st.image(img_tree_asli, caption="Struktur Percabangan Pohon Keputusan Nyata (Tree #1 dari 100 Estimators)", use_container_width=True)
                with open(img_tree_asli, "rb") as f_tree:
                    st.download_button("📥 Unduh Diagram Pohon #1 (Resolusi Tinggi 300 DPI)", data=f_tree, file_name="struktur_pohon_keputusan_tree1.png", mime="image/png")
                    st.download_button("Unduh Diagram Pohon #1 (PNG)", data=f_tree, file_name="struktur_pohon_keputusan_tree1.png", mime="image/png")

        st.divider()
        st.subheader("📊 Hasil Komputasi Voting Nyata 100 Pohon Keputusan")
        st.markdown(
            "Grafik ini menyajikan **akumulasi suara nyata dari seluruh 100 pohon Random Forest** untuk setiap siswa data uji: "
            "memperlihatkan sebaran suara pohon (0 s/d 100 suara) per rumpun prodi dan tingkat kebulatan konsensus (*agreement level*)."
        )
        st.subheader("Hasil Komputasi Voting Nyata 100 Pohon Keputusan")
        st.caption("Distribusi sebaran suara 100 pohon keputusan pada 36 siswa data uji independen.")
        
        img_voting = os.path.join(DIR_MODEL, "hasil_voting_nyata_100_pohon_177.png")
        if os.path.exists(img_voting):
            st.image(img_voting, caption="Distribusi Suara Nyata 100 Pohon & Tingkat Konsensus", use_container_width=True)
            st.image(img_voting, caption="Distribusi Suara 100 Pohon & Tingkat Konsensus Data Uji", use_container_width=True)
            with open(img_voting, "rb") as f_v:
                st.download_button(
                    label="📥 Unduh Grafik Voting 100 Pohon (Resolusi Tinggi 300 DPI)",
                    label="Unduh Grafik Voting 100 Pohon (Resolusi Tinggi)",
                    data=f_v,
                    file_name="hasil_voting_100_pohon.png",
                    mime="image/png"
                )
        
        # Tabel Numerik Suara Nyata
        csv_v = os.path.join(DIR_MODEL, "rekapitulasi_voting_100_pohon_177.csv")
        if os.path.exists(csv_v):
            st.markdown("#### Tabel Numerik Suara 100 Pohon per Siswa Uji:")
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("<b>Tabel Data Suara 100 Pohon per Siswa Uji:</b>", unsafe_allow_html=True)
            df_v_show = pd.read_csv(csv_v)
            st.dataframe(df_v_show, use_container_width=True)
            st.dataframe(df_v_show, use_container_width=True, hide_index=True)

# ==============================================================================
# MENU 3: DATA HASIL DIAGNOSA
# ==============================================================================
elif menu == "📑 Data Hasil Diagnosa":
    st.header("📑 Tabel Data Hasil Diagnosa & Prediksi")
    st.markdown("Rekapitulasi data siswa lengkap dengan status kesesuaian dan rekomendasi program studi.")
    st.divider()
elif menu == "Data Hasil Diagnosa":
    st.title("Data Hasil Diagnosa dan Prediksi")
    st.markdown("Tabel rekapitulasi data siswa lengkap dengan hasil diagnosa kesesuaian dan rekomendasi program studi.")
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if df_pred is not None:
        st.subheader("Tabel Hasil Prediksi pada Data Uji (Testing Set):")
        st.dataframe(df_pred, use_container_width=True)
        st.subheader("Hasil Prediksi pada Data Uji Independen (Holdout Test Set)")
        st.dataframe(df_pred, use_container_width=True, hide_index=True)
        csv_bytes = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Unduh Hasil Prediksi (CSV)",
            "Unduh Hasil Prediksi (CSV)",
            data=csv_bytes,
            file_name="hasil_rekomendasi_nalara.csv",
            mime="text/csv"
        )
    else:
        st.dataframe(df_data[['ID_Siswa', 'Jurusan', 'Status_Kesesuaian', 'Target_Rumpun_Kuliah']].head(20))
        st.dataframe(df_data[['ID_Siswa', 'Jurusan', 'Status_Kesesuaian', 'Target_Rumpun_Kuliah']].head(20), hide_index=True)

# ==============================================================================
# MENU 4: TENTANG SISTEM NALARA
# MENU 4: DOKUMENTASI METODOLOGI
# ==============================================================================
elif menu == "ℹ️ Tentang Sistem NALARA":
    st.header("ℹ️ Tentang Sistem NALARA")
elif menu == "Dokumentasi Metodologi":
    st.title("Dokumentasi Sistem NALARA")
    st.markdown(
        "Sistem Pendukung Keputusan Penentuan Arah Studi Lanjut bagi Siswa SMK Menggunakan "
        "Pendekatan Hibrida K-Means Clustering 2D dan Random Forest Classifier."
    )
    st.divider()

    st.markdown("""
    **NALARA (Nalar & Arah Karier)** adalah sistem pendukung keputusan penentuan arah studi lanjut bagi siswa SMK 
    yang mengintegrasikan data nilai akademik rapor dengan asesmen psikometri minat Holland RIASEC.
    ### 1. Landasan Metodologis: Two-Stage Hybrid Machine Learning
    1. **Tahap 1: Diagnosa Kongruensi (K-Means Clustering 2D, K=2)**  
       Mengevaluasi keselarasan antara kemampuan akademik rapor (Sumbu Y) dengan minat kejuruan saat ini (Sumbu X).  
       *Keluaran:* Status kesesuaian (1 = Kongruen / Sejalur, 0 = Diskongruen / Butuh Penyelamatan).
    2. **Tahap 2: Proyeksi Rekomendasi (Random Forest Classifier, 100 Trees)**  
       Mengintegrasikan 11 fitur masukan terstandarisasi (4 nilai rapor, 6 dimensi minat RIASEC, dan 1 status kesesuaian K-Means) 
       untuk memprediksi 5 rumpun program studi perguruan tinggi berstandar SNPMB.  
       *Keluaran:* Rekomendasi Linier (untuk siswa kongruen) dan Rekomendasi Penyelamatan (untuk siswa diskongruen).

    ### 🏗️ Metodologi: Two-Stage Hybrid Machine Learning
    1. **Tahap 1 (K-Means Clustering 2D, $K=2$):**  
       Mendiagnosa kesesuaian antara kemampuan kognitif akademik (Sumbu $Y$) dengan minat kejuruan saat ini (Sumbu $X$).  
       *Hasil:* Label `Status_Kesesuaian` (1 = Sesuai, 0 = Mismatch).
    2. **Tahap 2 (Random Forest Classifier):**  
       Menggabungkan 11 fitur (4 Rapor + 6 RIASEC + 1 Status Kesesuaian K-Means) untuk memprediksi rekomendasi 5 Rumpun Program Studi PT.  
       *Hasil:* Rekomendasi Linier (bagi siswa sesuai) dan Rekomendasi Penyelamatan (bagi siswa mismatch).

    ### 🏫 Lokasi & Subjek Studi Kasus
    * **Sekolah:** SMK Negeri 1 Bintan Timur, Kepulauan Riau
    * **Komposisi 5 Jurusan:** RPL, AKL, PH, DKV, dan TB (Populasi Siswa Aktif).
    ### 2. Karakteristik Subjek Penelitian
    * **Lokasi Riset:** SMK Negeri 1 Bintan Timur, Kepulauan Riau
    * **Konsentrasi Keahlian:** Rekayasa Perangkat Lunak (RPL), Akuntansi (AKT), Perhotelan (PH), Desain Komunikasi Visual (DKV), dan Tata Busana (TB).
    * **Sumber Data:** 29 berkas buku leger rapor semester genap resmi dan kuesioner minat RIASEC.
    """)
    st.info("Sistem ini dibangun untuk memenuhi Laporan Praktikum Mata Kuliah Penambangan Data (Semester 5).")

