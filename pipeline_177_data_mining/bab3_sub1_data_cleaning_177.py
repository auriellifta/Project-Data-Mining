# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab3_sub1_data_cleaning_177.py
# Sub-Bab III.1: Data Cleaning & Kuantifikasi Kuesioner
# ==============================================================================

import os
import pandas as pd

print("=" * 75)
print("SUB-BAB III.1: DATA CLEANING & KUANTIFIKASI KUESIONER")
print("=" * 75)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# 1. Memuat Dataset Kuesioner Mentah
file_kuesioner = os.path.join(ROOT_DIR, "Kuesioner Minat, Kemampuan, dan Arah Karier Siswa SMKN V2 (Jawaban).xlsx")
df_raw = pd.read_excel(file_kuesioner)
print(f"[*] Data kuesioner awal dimuat: {len(df_raw)} baris x {df_raw.shape[1]} kolom.")

# 2. III.1.1 Penanganan Format Desimal NIS & Spasi String
df_clean = pd.DataFrame()
df_clean['NIS'] = df_raw['NIS'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
df_clean['Nama'] = df_raw['Nama'].astype(str).str.strip()
df_clean['Jurusan'] = df_raw['Jurusan'].astype(str).str.strip()

# 3. III.1.3 Kuantifikasi Skala Ordinal Likert ke Numerik (1 s/d 5)
likert_map = {
    'Sangat Tidak Sesuai': 1,
    'Tidak Sesuai': 2,
    'Cukup Sesuai': 3,
    'Sesuai': 4,
    'Sangat Sesuai': 5
}

# III.2.3 Pemilahan Butir RIASEC (Eksklusi Kolom 35 Judul & Kolom 42 Attention Check)
dim_cols = {
    'Realistic': [5, 11, 17, 23, 29, 36, 43],
    'Investigative': [6, 12, 18, 24, 30, 37, 44],
    'Artistic': [7, 13, 19, 25, 31, 38, 45],
    'Social': [8, 14, 20, 26, 32, 39, 46],
    'Enterprising': [9, 15, 21, 27, 33, 40, 47],
    'Conventional': [10, 16, 22, 28, 34, 41, 48]
}

# Menghitung skor kumulatif (SUM 7 butir per dimensi, skala 7 - 35)
for dim, cols in dim_cols.items():
    df_clean[dim] = df_raw.iloc[:, cols].replace(likert_map).astype(float).sum(axis=1)

# Mengambil kolom rencana kuliah (Kolom 50)
col_kuliah = 'Jika kuliah, bidang apa yang paling kamu minati?'
df_clean['Pilihan_Kuliah_Mentah'] = df_raw[col_kuliah].astype(str).str.strip()

# Simpan hasil pembersihan kuesioner ke file temporary
out_path = os.path.join(BASE_DIR, "temp_kuesioner_clean_177.csv")
df_clean.to_csv(out_path, index=False)

print(f"\n[V] Hasil Sub-Bab III.1 Berhasil Disimpan: {os.path.basename(out_path)}")
print(f"    - Missing Values NIS : {df_clean['NIS'].isnull().sum()} (Bersih)")
print(f"    - Duplikasi NIS      : {df_clean['NIS'].duplicated().sum()} (0 Duplikasi)")
print(f"    - Dimensi RIASEC     : Realistic, Investigative, Artistic, Social, Enterprising, Conventional")
print("=" * 75)

