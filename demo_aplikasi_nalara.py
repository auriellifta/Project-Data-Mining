# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: demo_aplikasi_nalara.py
# Deskripsi: Program Demo Implementasi Sistem NALARA (Inference Engine)
#            Menerapkan Two-Stage Hybrid (K-Means 2D & Random Forest Classifier)
# ==============================================================================

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "pipeline_177_data_mining", "output_model")
DATA_DIR = os.path.join(BASE_DIR, "pipeline_177_data_mining")

# 1. Memuat Model Terlatih dan Dataset Acuan
path_rf_model = os.path.join(MODEL_DIR, "model_random_forest_177.joblib")
path_df = os.path.join(DATA_DIR, "dataset_terklaster_177.csv")

rf_model = joblib.load(path_rf_model)
df_ref = pd.read_csv(path_df)

# Menghitung parameter statistik Mean dan Std dari data latih untuk normalisasi Z-score
fitur_numerik = [
    'Matematika', 'Bahasa_Indonesia', 'Bahasa_Inggris', 'Produktif', 'Rata_Rapor',
    'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional', 'Skor_Minat_Kejuruan'
]
means = df_ref[fitur_numerik].mean()
stds = df_ref[fitur_numerik].std()

# Titik Centroid K-Means 2D (Skala Z-score)
c0_scaled = np.array([-0.82320427, 0.15817904])  # Mismatch (Cluster 0)
c1_scaled = np.array([0.77797326, -0.14948789])  # Sesuai (Cluster 1)

def inferensi_siswa(nama_siswa, jurusan_smk, nilai_rapor, skor_riasec):
    """
    Fungsi Inferensi Lengkap Two-Stage Hybrid NALARA
    """
    mat, bi, en, prod = nilai_rapor
    r, i, a, s, e, c = skor_riasec
    rata_rapor = (mat + bi + en + prod) / 4.0
    
    # 1. Tentukan Skor Minat Kejuruan Sumbu X K-Means
    jur = jurusan_smk.upper()
    if 'RPL' in jur or 'REKAYASA' in jur:
        minat_kejuruan = i
        dim_kejuruan = 'Investigative (RPL)'
    elif 'AKL' in jur or 'AKUNTANSI' in jur:
        minat_kejuruan = c
        dim_kejuruan = 'Conventional (AKL)'
    elif 'PH' in jur or 'PERHOTELAN' in jur:
        minat_kejuruan = s
        dim_kejuruan = 'Social (Perhotelan)'
    elif 'DKV' in jur or 'DESAIN' in jur:
        minat_kejuruan = a
        dim_kejuruan = 'Artistic (DKV)'
    elif 'TB' in jur or 'BUSANA' in jur:
        minat_kejuruan = a
        dim_kejuruan = 'Artistic (Tata Busana)'
    else:
        minat_kejuruan = r
        dim_kejuruan = 'Realistic'
        
    # 2. Normalisasi Z-Score untuk Sumbu X dan Sumbu Y K-Means
    x_scaled = (minat_kejuruan - means['Skor_Minat_Kejuruan']) / stds['Skor_Minat_Kejuruan']
    y_scaled = (rata_rapor - means['Rata_Rapor']) / stds['Rata_Rapor']
    
    # 3. Diagnosa Stage 1: K-Means Clustering 2D
    d0 = np.sqrt((x_scaled - c0_scaled[0])**2 + (y_scaled - c0_scaled[1])**2)
    d1 = np.sqrt((x_scaled - c1_scaled[0])**2 + (y_scaled - c1_scaled[1])**2)
    
    if d1 < d0:
        klaster_kmeans = 1
        status_teks = "Sesuai Jurusan (High Congruence)"
        catatan_diagnosa = "Profil minat kejuruan siswa selaras dengan jurusan SMK yang ditempuh saat ini."
    else:
        klaster_kmeans = 0
        status_teks = "Tidak Sesuai / Mismatch (Potential Need for Rescue)"
        catatan_diagnosa = "Profil minat kejuruan siswa berada di bawah ambang kongruen; membutuhkan rekomendasi penyelamatan lintas bidang ke perguruan tinggi."
        
    # 4. Normalisasi 11 Fitur Masukan untuk Stage 2 (Random Forest)
    z_mat = (mat - means['Matematika']) / stds['Matematika']
    z_bi = (bi - means['Bahasa_Indonesia']) / stds['Bahasa_Indonesia']
    z_en = (en - means['Bahasa_Inggris']) / stds['Bahasa_Inggris']
    z_prod = (prod - means['Produktif']) / stds['Produktif']
    z_r = (r - means['Realistic']) / stds['Realistic']
    z_i = (i - means['Investigative']) / stds['Investigative']
    z_a = (a - means['Artistic']) / stds['Artistic']
    z_s = (s - means['Social']) / stds['Social']
    z_e = (e - means['Enterprising']) / stds['Enterprising']
    z_c = (c - means['Conventional']) / stds['Conventional']
    
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
    
    # 5. Prediksi Rekomendasi Rumpun PT Stage 2
    pred_prodi = rf_model.predict(X_input)[0]
    proba = rf_model.predict_proba(X_input)[0]
    kelas_rf = rf_model.classes_
    top_confidence = np.max(proba) * 100
    
    # Identifikasi Tipe Rekomendasi (Linier vs Penyelamatan)
    if klaster_kmeans == 1:
        tipe_rekomendasi = "Rekomendasi Linier (Sejalur dengan Jurusan SMK)"
    else:
        tipe_rekomendasi = "Rekomendasi Penyelamatan (Lintas Minat Psikometri Sejati)"
        
    return {
        'Nama': nama_siswa,
        'Jurusan': jurusan_smk,
        'Rata_Rapor': rata_rapor,
        'Skor_Minat_Kejuruan': minat_kejuruan,
        'Dimensi_Kejuruan': dim_kejuruan,
        'Jarak_C0': d0,
        'Jarak_C1': d1,
        'Status_Diagnosa': status_teks,
        'Catatan_Diagnosa': catatan_diagnosa,
        'Rekomendasi_Rumpun_PT': pred_prodi,
        'Tipe_Rekomendasi': tipe_rekomendasi,
        'Confidence': top_confidence,
        'Detail_Probabilitas': dict(zip(kelas_rf, proba * 100))
    }

print("=" * 85)
print("DEMO SISTEM NALARA - INFERENCE ENGINE TWO-STAGE HYBRID MACHINE LEARNING")
print("Studi Kasus: SMKN 1 Bintan Timur | Model: K-Means 2D + Random Forest Classifier")
print("=" * 85)

# Uji Coba Skenario 1: Siswa RPL Sesuai Jurusan
siswa_1 = inferensi_siswa(
    nama_siswa="Ahmad Fauzan (Simulasi Siswa A)",
    jurusan_smk="Rekayasa Perangkat Lunak (RPL)",
    nilai_rapor=[85.0, 84.0, 86.0, 90.0],  # Mat, Indo, Ing, Produktif
    skor_riasec=[24, 32, 18, 20, 22, 26]   # R, I, A, S, E, C (Investigative tinggi = 32)
)

print(f"\n[HASIL INFERENSI KASUS 1 - SISWA SESUAI]")
print(f" * Nama Siswa           : {siswa_1['Nama']}")
print(f" * Jurusan Asal SMK     : {siswa_1['Jurusan']}")
print(f" * Rata-Rata 4 Mapel    : {siswa_1['Rata_Rapor']:.2f} (Skala 0 - 100)")
print(f" * Minat Kejuruan       : {siswa_1['Skor_Minat_Kejuruan']} ({siswa_1['Dimensi_Kejuruan']})")
print(f" --- [TAHAP 1: K-MEANS 2D DIAGNOSA AWAL] ---")
print(f" * Jarak ke Centroid 0  : {siswa_1['Jarak_C0']:.4f} (Mismatch)")
print(f" * Jarak ke Centroid 1  : {siswa_1['Jarak_C1']:.4f} (Sesuai)")
print(f" * Diagnosa Kesesuaian  : {siswa_1['Status_Diagnosa']}")
print(f" --- [TAHAP 2: RANDOM FOREST REKOMENDASI PRODI PT] ---")
print(f" * Rekomendasi Rumpun PT: >>> {siswa_1['Rekomendasi_Rumpun_PT']} <<<")
print(f" * Sifat Rekomendasi    : {siswa_1['Tipe_Rekomendasi']}")
print(f" * Tingkat Keyakinan    : {siswa_1['Confidence']:.2f}%")

# Uji Coba Skenario 2: Siswa Akuntansi Mengalami Mismatch (Minat Seni Menonjol)
siswa_2 = inferensi_siswa(
    nama_siswa="Siti Rahmawati (Simulasi Siswa B)",
    jurusan_smk="Akuntansi (AKT)",
    nilai_rapor=[80.0, 82.0, 84.0, 82.0],  # Mat, Indo, Ing, Produktif
    skor_riasec=[19, 21, 33, 23, 20, 22]   # R, I, A, S, E, C (Conventional rendah = 22, Artistic tinggi = 33)
)

print(f"\n[HASIL INFERENSI KASUS 2 - SISWA MISMATCH DENGAN REKOMENDASI PENYELAMATAN]")
print(f" * Nama Siswa           : {siswa_2['Nama']}")
print(f" * Jurusan Asal SMK     : {siswa_2['Jurusan']}")
print(f" * Rata-Rata 4 Mapel    : {siswa_2['Rata_Rapor']:.2f} (Skala 0 - 100)")
print(f" * Minat Kejuruan       : {siswa_2['Skor_Minat_Kejuruan']} ({siswa_2['Dimensi_Kejuruan']})")
print(f" --- [TAHAP 1: K-MEANS 2D DIAGNOSA AWAL] ---")
print(f" * Jarak ke Centroid 0  : {siswa_2['Jarak_C0']:.4f} (Mismatch)")
print(f" * Jarak ke Centroid 1  : {siswa_2['Jarak_C1']:.4f} (Sesuai)")
print(f" * Diagnosa Kesesuaian  : {siswa_2['Status_Diagnosa']}")
print(f" * Catatan Sistem       : {siswa_2['Catatan_Diagnosa']}")
print(f" --- [TAHAP 2: RANDOM FOREST REKOMENDASI PRODI PT] ---")
print(f" * Rekomendasi Rumpun PT: >>> {siswa_2['Rekomendasi_Rumpun_PT']} <<<")
print(f" * Sifat Rekomendasi    : {siswa_2['Tipe_Rekomendasi']}")
print(f" * Tingkat Keyakinan    : {siswa_2['Confidence']:.2f}%")
print("=" * 85)

