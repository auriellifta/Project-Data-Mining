# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab4_sub2_random_forest_177.py
# Sub-Bab IV.2: Pemodelan Stage 2 - Random Forest Classifier (11 Fitur Masukan)
# ==============================================================================

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("=" * 85)
print("SUB-BAB IV.2: PEMODELAN STAGE 2 - RANDOM FOREST CLASSIFIER")
print("=" * 85)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Memuat Dataset Berlabel Diagnosa Hasil Stage 1 K-Means (dataset_terklaster_177.csv)
path_data = os.path.join(BASE_DIR, "dataset_terklaster_177.csv")
df = pd.read_csv(path_data)
print(f"[*] Dataset Terklaster Dimuat : {len(df)} baris x {df.shape[1]} kolom.")

# 2. Menyiapkan 11 Fitur Masukan (Features X) dan Label Target (Target Y)
# 4 Rapor + 6 RIASEC + 1 Status_Kesesuaian K-Means
feature_cols = [
    'Matematika_scaled', 'Bahasa_Indonesia_scaled', 'Bahasa_Inggris_scaled', 'Produktif_scaled',
    'Realistic_scaled', 'Investigative_scaled', 'Artistic_scaled', 'Social_scaled', 'Enterprising_scaled', 'Conventional_scaled',
    'Klaster_KMeans'
]

nama_fitur_tampil = [
    'Matematika (Z)', 'B. Indonesia (Z)', 'B. Inggris (Z)', 'Produktif (Z)',
    'Realistic (Z)', 'Investigative (Z)', 'Artistic (Z)', 'Social (Z)', 'Enterprising (Z)', 'Conventional (Z)',
    'Status_Kesesuaian_KMeans (0/1)'
]

X = df[feature_cols]
y = df['Target_Rumpun_Kuliah']

# 3. Pembagian Data Latih dan Data Uji (Stratified Split 80:20)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.20, random_state=42, stratify=y
)

print(f"[*] Pembagian Data :")
print(f"    - Data Latih (Training 80%) : {len(X_train)} siswa")
print(f"    - Data Uji   (Testing  20%) : {len(X_test)} siswa")
print(f"    - Jumlah Fitur Masukan      : {X.shape[1]} fitur")
print(f"    - Kelas Target Rumpun PT   : {len(np.unique(y))} kelas (5 Rumpun SNPMB)")

# 4. Inisialisasi & Pelatihan Model Random Forest Classifier
# Hyperparameter: 100 Decision Trees, kriteria Gini Impurity, random_state=42
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

print("\n[*] Sedang melatih 100 Decision Trees pada data latih...")
rf_model.fit(X_train, y_train)
print("[V] Pelatihan Random Forest Selesai 100%!")

# 5. Evaluasi Cepat Performa Pelatihan
train_acc = rf_model.score(X_train, y_train)
test_acc = rf_model.score(X_test, y_test)
print(f"\n[*] Tingkat Akurasi Model:")
print(f"    - Akurasi Data Latih (Training) : {train_acc * 100:.2f}%")
print(f"    - Akurasi Data Uji   (Testing)  : {test_acc * 100:.2f}%")

# 6. Menyimpan Model Terlatih & Hasil Prediksi Data Uji
path_model_save = os.path.join(OUTPUT_DIR, "model_random_forest_177.joblib")
joblib.dump(rf_model, path_model_save)
print(f"[V] File Model Tersimpan: {os.path.basename(path_model_save)}")

# Membuat tabel hasil prediksi pada data uji
df_test_result = df.loc[idx_test, ['ID_Siswa', 'Jurusan', 'Status_Kesesuaian', 'Target_Rumpun_Kuliah']].copy()
df_test_result['Prediksi_Random_Forest'] = rf_model.predict(X_test)
proba = rf_model.predict_proba(X_test)
df_test_result['Confidence_Score (%)'] = np.max(proba, axis=1) * 100
df_test_result['Status_Prediksi'] = np.where(
    df_test_result['Target_Rumpun_Kuliah'] == df_test_result['Prediksi_Random_Forest'],
    'BENAR', 'SALAH'
)

path_pred_save = os.path.join(BASE_DIR, "hasil_prediksi_rf_177.csv")
df_test_result.to_csv(path_pred_save, index=False)
print(f"[V] Tabel Prediksi Disimpan : {os.path.basename(path_pred_save)}")

# 7. Demonstrasi 2 Kasus Uji Nyata: Siswa Sesuai (Linier) vs Siswa Mismatch (Penyelamatan)
print("\n" + "=" * 85)
print("DEMONSTRASI PENGUJIAN MODEL PADA DATA UJI (DUAL RECOMENDATION SCENARIO):")
print("=" * 85)

# Cari sampel Siswa Sesuai yang prediksinya Benar
sampel_sesuai = df_test_result[(df_test_result['Status_Kesesuaian'] == 'Sesuai Jurusan') & (df_test_result['Status_Prediksi'] == 'BENAR')].iloc[0]
print(f"[Kasus 1 - Rekomendasi Linier Sejalur]")
print(f" * ID Siswa             : {sampel_sesuai['ID_Siswa']}")
print(f" * Jurusan SMK Asal     : {sampel_sesuai['Jurusan']}")
print(f" * Diagnosa K-Means     : {sampel_sesuai['Status_Kesesuaian']}")
print(f" * Target Aktual        : {sampel_sesuai['Target_Rumpun_Kuliah']}")
print(f" * Rekomendasi Model RF : {sampel_sesuai['Prediksi_Random_Forest']}")
print(f" * Tingkat Keyakinan    : {sampel_sesuai['Confidence_Score (%)']:.2f}% (Keputusan Linier)")

# Cari sampel Siswa Mismatch yang prediksinya Benar (Penyelamatan Lintas Jalur)
sampel_mismatch = df_test_result[(df_test_result['Status_Kesesuaian'] == 'Tidak Sesuai (Mismatch)') & (df_test_result['Status_Prediksi'] == 'BENAR')].iloc[0]
print(f"\n[Kasus 2 - Rekomendasi Penyelamatan Lintas Minat]")
print(f" * ID Siswa             : {sampel_mismatch['ID_Siswa']}")
print(f" * Jurusan SMK Asal     : {sampel_mismatch['Jurusan']}")
print(f" * Diagnosa K-Means     : {sampel_mismatch['Status_Kesesuaian']}")
print(f" * Target Aktual        : {sampel_mismatch['Target_Rumpun_Kuliah']}")
print(f" * Rekomendasi Model RF : {sampel_mismatch['Prediksi_Random_Forest']}")
print(f" * Tingkat Keyakinan    : {sampel_mismatch['Confidence_Score (%)']:.2f}% (Keputusan Penyelamatan)")
print("=" * 85)

