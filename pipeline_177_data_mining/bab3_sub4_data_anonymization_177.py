# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab3_sub4_data_anonymization_177.py
# Sub-Bab III.4: Data Anonymization (Pembersihan PII & Pseudonimisasi Siswa)
# ==============================================================================

import os
import pandas as pd

print("=" * 80)
print("SUB-BAB III.4: DATA ANONYMIZATION (PROTEKSI PRIVASI DATA SISWA)")
print("=" * 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Memuat Dataset Terintegrasi dari Sub-Bab III.3
path_terintegrasi = os.path.join(BASE_DIR, "dataset_terintegrasi_177.csv")
df_master = pd.read_csv(path_terintegrasi, dtype={'NIS': str})

print(f"[*] Dataset Terintegrasi Dimuat: {len(df_master)} siswa.")
print(f"[*] Kolom Terdeteksi Mengandung PII: ['NIS', 'Nama']")

# 2. Pengurutan Data & Pembangkitan Identitas Pseudonim (ID_Siswa)
df_master = df_master.sort_values(by='NIS').reset_index(drop=True)
df_master['ID_Siswa'] = [f"Siswa_{i+1:03d}" for i in range(len(df_master))]

# 3. Eliminasi (Drop) Atribut Identitas Pribadi (PII)
kolom_non_pii = [
    'ID_Siswa', 'Jurusan', 
    'Matematika', 'Bahasa_Indonesia', 'Bahasa_Inggris', 'Produktif',
    'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional',
    'Pilihan_Kuliah_Mentah'
]

df_anonim = df_master[kolom_non_pii].copy()

# 4. Menyimpan Dataset Anonim
out_path = os.path.join(BASE_DIR, "dataset_anonim_177.csv")
df_anonim.to_csv(out_path, index=False)

print(f"\n[V] Hasil Sub-Bab III.4 Berhasil Disimpan: {os.path.basename(out_path)}")
print(f"    - Kolom 'NIS' & 'Nama'       : Berhasil dihapus permanen dari dataset olahan")
print(f"    - Rentang ID Pengganti      : {df_anonim['ID_Siswa'].iloc[0]} s/d {df_anonim['ID_Siswa'].iloc[-1]}")
print(f"    - Dimensi Dataset Anonim    : {df_anonim.shape[0]} baris x {df_anonim.shape[1]} kolom")
print(f"    - Contoh 3 Baris Teratas:\n")
print(df_anonim[['ID_Siswa', 'Jurusan', 'Matematika', 'Produktif', 'Realistic', 'Pilihan_Kuliah_Mentah']].head(3).to_string(index=False))
print("=" * 80)

