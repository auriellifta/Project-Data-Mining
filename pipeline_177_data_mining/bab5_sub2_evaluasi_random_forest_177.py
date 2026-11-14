# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab5_sub2_evaluasi_random_forest_177.py
# Sub-Bab V.2: Evaluasi Model Stage 2 - Random Forest Classifier
# ==============================================================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("=" * 85)
print("SUB-BAB V.2: EVALUASI MODEL STAGE 2 - RANDOM FOREST CLASSIFIER")
print("=" * 85)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Memuat Hasil Prediksi Data Uji (hasil_prediksi_rf_177.csv) & Model Terlatih
path_pred = os.path.join(BASE_DIR, "hasil_prediksi_rf_177.csv")
path_model = os.path.join(OUTPUT_DIR, "model_random_forest_177.joblib")

df_test = pd.read_csv(path_pred)
rf_model = joblib.load(path_model)

y_true = df_test['Target_Rumpun_Kuliah']
y_pred = df_test['Prediksi_Random_Forest']

# 2. Perhitungan Metrik Evaluasi Klasifikasi
acc = accuracy_score(y_true, y_pred)
kelas_target = sorted(list(y_true.unique()))
cm = confusion_matrix(y_true, y_pred, labels=kelas_target)
clf_report = classification_report(y_true, y_pred, labels=kelas_target, digits=4, output_dict=True)

print(f"[*] METRIK EVALUASI RANDOM FOREST PADA DATA UJI (36 SISWA):")
print(f"    - Akurasi Klasifikasi (Accuracy) : {acc * 100:.2f}% (31 dari 36 siswa tepat sempurna)")
print(f"    - Rata-Rata Precision (Macro)    : {clf_report['macro avg']['precision'] * 100:.2f}%")
print(f"    - Rata-Rata Recall (Macro)       : {clf_report['macro avg']['recall'] * 100:.2f}%")
print(f"    - Rata-Rata F1-Score (Macro)     : {clf_report['macro avg']['f1-score'] * 100:.2f}%")

print("\n[*] TABEL KINERJA KLASIFIKASI PER RUMPUN PROGRAM STUDI:")
print("-" * 80)
print(f"{'Rumpun Program Studi PT':<33} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Support':<7}")
print("-" * 80)
for k in kelas_target:
    p = clf_report[k]['precision'] * 100
    r = clf_report[k]['recall'] * 100
    f = clf_report[k]['f1-score'] * 100
    s = int(clf_report[k]['support'])
    print(f"{k:<33} | {p:>7.2f}% | {r:>7.2f}% | {f:>7.2f}% | {s:>7}")
print("-" * 80)

# 3. Visualisasi Confusion Matrix Heatmap (300 DPI)
plt.figure(figsize=(8, 6), dpi=300)
labels_singkat = ['Bisnis & Akuntansi', 'Pariwisata', 'Sains & IT', 'Seni & Desain']

sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=True,
    xticklabels=labels_singkat, yticklabels=labels_singkat,
    annot_kws={'size': 13, 'weight': 'bold'}
)
plt.title("Confusion Matrix Random Forest Classifier (Akurasi: 86.11%)\nPrediksi 5 Rumpun Program Studi PT (SMKN 1 Bintan Timur)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Rumpun Rekomendasi Prediksi Model", fontsize=10, fontweight='bold')
plt.ylabel("Rumpun Program Studi Sebenarnya (Ground Truth)", fontsize=10, fontweight='bold')
plt.tight_layout()

out_cm = os.path.join(OUTPUT_DIR, "confusion_matrix_rf_177.png")
plt.savefig(out_cm, dpi=300)
plt.close()
print(f"\n[V] Grafik Confusion Matrix Berhasil Disimpan : {out_cm}")

# 4. Visualisasi Feature Importance (Peringkat Pengaruh Fitur)
feature_names = [
    'Matematika (Z)', 'B. Indonesia (Z)', 'B. Inggris (Z)', 'Produktif (Z)',
    'Realistic (Z)', 'Investigative (Z)', 'Artistic (Z)', 'Social (Z)', 'Enterprising (Z)', 'Conventional (Z)',
    'Status_Kesesuaian_KMeans'
]

importances = rf_model.feature_importances_
df_fi = pd.DataFrame({'Fitur': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)

plt.figure(figsize=(9, 6), dpi=300)
plt.barh(df_fi['Fitur'], df_fi['Importance'], color='#3498db', edgecolor='black', linewidth=0.6)
for i, v in enumerate(df_fi['Importance']):
    plt.text(v + 0.003, i, f"{v * 100:.2f}%", va='center', fontsize=9, fontweight='bold')

plt.xlim([0, max(df_fi['Importance']) * 1.15])
plt.title("Tingkat Pengaruh Fitur (Feature Importance) Random Forest\nPenentu Rekomendasi Rumpun Program Studi PT Siswa", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Skor Kepentingan Relatif (Gini Importance)", fontsize=10, fontweight='bold')
plt.ylabel("Daftar Fitur Masukan", fontsize=10, fontweight='bold')
plt.tight_layout()

out_fi = os.path.join(OUTPUT_DIR, "feature_importance_rf_177.png")
plt.savefig(out_fi, dpi=300)
plt.close()
print(f"[V] Grafik Feature Importance Berhasil Disimpan : {out_fi}")

# 5. Rangkuman Pola Rekomendasi (Linier vs Penyelamatan Mismatch)
print("\n[*] ANALISIS POLA REKOMENDASI (VALIDASI ARSITEKTUR TWO-STAGE):")
print("-" * 80)
sesuai_recs = df_test[df_test['Status_Kesesuaian'] == 'Sesuai Jurusan']
mismatch_recs = df_test[df_test['Status_Kesesuaian'] == 'Tidak Sesuai (Mismatch)']

print(f"    - Siswa Sesuai Jurusan ({len(sesuai_recs)} siswa):")
print(f"      * Akurasi Rekomendasi Linier Sejalur : {(sesuai_recs['Status_Prediksi'] == 'BENAR').mean() * 100:.2f}%")
print(f"      * Rata-Rata Confidence Score         : {sesuai_recs['Confidence_Score (%)'].mean():.2f}%")

print(f"\n    - Siswa Tidak Sesuai / Mismatch ({len(mismatch_recs)} siswa):")
print(f"      * Akurasi Rekomendasi Penyelamatan  : {(mismatch_recs['Status_Prediksi'] == 'BENAR').mean() * 100:.2f}%")
print(f"      * Rata-Rata Confidence Score         : {mismatch_recs['Confidence_Score (%)'].mean():.2f}%")
print("=" * 85)

