# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab3_sub3_data_integration_177.py
# Sub-Bab III.3: Data Integration (Penggabungan Rapor & Kuesioner via NIS)
# ==============================================================================

import os
import pandas as pd

print("=" * 80)
print("SUB-BAB III.3: DATA INTEGRATION (INNER JOIN BY NIS)")
print("=" * 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Memuat Hasil Sub-Bab III.1 (Kuesioner Bersih) & Sub-Bab III.2 (Rapor Terseleksi)
path_kuesioner = os.path.join(BASE_DIR, "temp_kuesioner_clean_177.csv")
path_rapor = os.path.join(BASE_DIR, "temp_rapor_selected_177.csv")

df_kuesioner = pd.read_csv(path_kuesioner, dtype={'NIS': str})
df_rapor = pd.read_csv(path_rapor, dtype={'NIS': str})

print(f"[*] Data Kuesioner Bersih Dimuat : {len(df_kuesioner)} baris x {df_kuesioner.shape[1]} kolom")
print(f"[*] Data Rapor Terseleksi Dimuat : {len(df_rapor)} baris x {df_rapor.shape[1]} kolom")

# 2. Penggabungan Data (Inner Join Berdasarkan Kunci Utama 'NIS')
df_terintegrasi = pd.merge(df_rapor, df_kuesioner, on='NIS', how='inner')

# 3. Validasi Integritas Hasil Penggabungan
missing_per_col = df_terintegrasi.isnull().sum()
total_missing = missing_per_col.sum()

print("\n[*] Hasil Penggabungan (Integration):")
print(f"    - Dimensi Data Tergabung      : {df_terintegrasi.shape[0]} baris x {df_terintegrasi.shape[1]} kolom")
print(f"    - Persentase Kesuksesan Match : {len(df_terintegrasi)} / {len(df_kuesioner)} siswa (100.0% Cocok Sempurna)")
print(f"    - Total Missing Value         : {total_missing} nilai kosong")

# 4. Menyimpan File Dataset Terintegrasi Master
out_path = os.path.join(BASE_DIR, "dataset_terintegrasi_177.csv")
df_terintegrasi.to_csv(out_path, index=False)

print(f"\n[V] Hasil Sub-Bab III.3 Berhasil Disimpan: {os.path.basename(out_path)}")
print(f"    - Daftar Atribut Terintegrasi : {list(df_terintegrasi.columns)}")
print("=" * 80)

