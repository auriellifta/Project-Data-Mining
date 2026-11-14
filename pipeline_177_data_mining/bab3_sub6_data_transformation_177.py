# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab3_sub6_data_transformation_177.py
# Sub-Bab III.6: Data Transformation (Standardisasi Z-Score StandardScaler)
# ==============================================================================

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("=" * 85)
print("SUB-BAB III.6: DATA TRANSFORMATION (STANDARDISASI FITUR Z-SCORE)")
print("=" * 85)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Memuat Dataset Ter-engineer dari Sub-Bab III.5
path_engineered = os.path.join(BASE_DIR, "dataset_engineered_177.csv")
df = pd.read_csv(path_engineered)

# 2. Pemilahan 12 Fitur Numerik yang Memerlukan Penyetaraan Skala
# (Nilai Rapor berskala 0-100 dan Skor RIASEC berskala 7-35)
fitur_numerik = [
    'Matematika', 'Bahasa_Indonesia', 'Bahasa_Inggris', 'Produktif', 'Rata_Rapor',
    'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional', 'Skor_Minat_Kejuruan'
]

print(f"[*] Jumlah Fitur Numerik yang Ditransformasi : {len(fitur_numerik)} fitur")
print(f"[*] Skala Awal Rapor (0 - 100) vs Skala Awal RIASEC (7 - 35)")

# 3. Penerapan Standardisasi Z-Score (Mean = 0, Std = 1)
# Rumus: Z = (X - Mean) / Std
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[fitur_numerik])
df_scaled = pd.DataFrame(X_scaled, columns=[f"{c}_scaled" for c in fitur_numerik])

# 4. Penggabungan Fitur Terstandarisasi dengan Informasi Identitas & Target
df_siap_model = pd.concat([
    df[['ID_Siswa', 'Jurusan', 'Target_Rumpun_Kuliah']], 
    df[fitur_numerik],
    df_scaled
], axis=1)

# 5. Menyimpan Dataset Akhir yang Siap untuk Pemodelan Stage 1 & Stage 2
out_path = os.path.join(BASE_DIR, "dataset_siap_model_177.csv")
df_siap_model.to_csv(out_path, index=False)

print(f"\n[V] Hasil Sub-Bab III.6 Berhasil Disimpan: {os.path.basename(out_path)}")
print(f"    - Total Kolom Dataset Siap Model : {df_siap_model.shape[1]} kolom")
print(f"    - Verifikasi Rata-Rata Skala Z   : Mean ~ {df_scaled.mean().mean():.4f} (Mendekati 0)")
print(f"    - Verifikasi Standar Deviasi Z  : Std  ~ {df_scaled.std().mean():.4f} (Mendekati 1)")
print("\n[*] Contoh 3 Baris Hasil Transformasi (Sebelum vs Sesudah Scaling):")
contoh_tampil = df_siap_model[['ID_Siswa', 'Rata_Rapor', 'Rata_Rapor_scaled', 'Skor_Minat_Kejuruan', 'Skor_Minat_Kejuruan_scaled']].head(3)
print(contoh_tampil.to_string(index=False))
print("=" * 85)

