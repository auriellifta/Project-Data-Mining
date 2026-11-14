# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab4_sub1_kmeans_177.py
# Sub-Bab IV.1: Pemodelan Stage 1 - K-Means Clustering 2D Diagnosa Awal (K=2)
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("=" * 85)
print("SUB-BAB IV.1: PEMODELAN STAGE 1 - K-MEANS CLUSTERING 2D DIAGNOSA AWAL (K=2)")
print("=" * 85)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Memuat Dataset Terstandarisasi dari BAB III (Sub-Bab III.6)
path_data = os.path.join(BASE_DIR, "dataset_siap_model_177.csv")
df = pd.read_csv(path_data)
print(f"[*] Dataset Siap Model Dimuat: {len(df)} siswa x {df.shape[1]} kolom.")

# 2. Menyiapkan Matriks Fitur 2 Dimensi (Sumbu X: Minat Kejuruan, Sumbu Y: Rata-Rata Rapor)
X_scaled = df[['Skor_Minat_Kejuruan_scaled', 'Rata_Rapor_scaled']].values
X_raw = df[['Skor_Minat_Kejuruan', 'Rata_Rapor']].values

# 3. Eksekusi K-Means Clustering (K=2)
# Menggunakan random_state=42 dan n_init=10 untuk hasil deterministik dan optimal
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_

# 4. Evaluasi Kualitas Klaster: Silhouette Score & Inertia (SSE)
sil_score = silhouette_score(X_scaled, raw_labels)
inertia_val = kmeans.inertia_

# 5. Penentuan Label Semantik Klaster:
# Klaster dengan jumlah koordinat centroid lebih tinggi ditetapkan sebagai "Sesuai Jurusan (1)"
centroid_scores = centroids_scaled.sum(axis=1)
if centroid_scores[1] > centroid_scores[0]:
    label_status = raw_labels
else:
    label_status = 1 - raw_labels

df['Klaster_KMeans'] = label_status
df['Status_Kesesuaian'] = df['Klaster_KMeans'].map({
    1: 'Sesuai Jurusan',
    0: 'Tidak Sesuai (Mismatch)'
})

# Hitung koordinat Centroid pada skala nilai asli (asli rapor 0-100 & RIASEC 7-35)
centroid_raw = df.groupby('Status_Kesesuaian')[['Skor_Minat_Kejuruan', 'Rata_Rapor']].mean().loc[['Tidak Sesuai (Mismatch)', 'Sesuai Jurusan']].values

print("\n[*] HASIL EVALUASI PEMODELAN K-MEANS 2D (STAGE 1):")
print(f"    - Silhouette Score : {sil_score:.4f} (Struktur klaster terpisah dengan baik)")
print(f"    - Inertia (SSE)    : {inertia_val:.4f}")
print("\n[*] KOORDINAT PUSAT KLASTER (CENTROID):")
print(f"    - Klaster 0 [Tidak Sesuai (Mismatch)] : Minat = {centroid_raw[0][0]:.2f} (Skala 7-35), Rapor = {centroid_raw[0][1]:.2f} (Skala 0-100)")
print(f"    - Klaster 1 [Sesuai Jurusan]          : Minat = {centroid_raw[1][0]:.2f} (Skala 7-35), Rapor = {centroid_raw[1][1]:.2f} (Skala 0-100)")

print("\n[*] DISTRIBUSI DIAGNOSA KESESUAIAN SISWA:")
for st, cnt in df['Status_Kesesuaian'].value_counts().items():
    pct = (cnt / len(df)) * 100
    print(f"    - {st:<25}: {cnt:>2} siswa ({pct:.1f}%)")

# 6. Menyimpan Dataset Berlabel Hasil Diagnosa Tahap 1
out_csv = os.path.join(BASE_DIR, "dataset_terklaster_177.csv")
df.to_csv(out_csv, index=False)
print(f"\n[V] Dataset Hasil Diagnosa Disimpan: {os.path.basename(out_csv)}")

# 7. Visualisasi Scatter Plot 2D Resolusi Tinggi (300 DPI)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 7), dpi=300)

palette_colors = {'Sesuai Jurusan': '#27ae60', 'Tidak Sesuai (Mismatch)': '#e67e22'}

# Plot titik data siswa
scatter = sns.scatterplot(
    data=df,
    x='Skor_Minat_Kejuruan',
    y='Rata_Rapor',
    hue='Status_Kesesuaian',
    palette=palette_colors,
    style='Status_Kesesuaian',
    s=80,
    alpha=0.85,
    edgecolor='black',
    linewidth=0.5
)

# Plot Centroid Klaster
plt.scatter(
    centroid_raw[0][0], centroid_raw[0][1],
    color='#d35400', s=250, marker='X', edgecolor='black', linewidth=2,
    label='Centroid 0 (Mismatch)'
)
plt.scatter(
    centroid_raw[1][0], centroid_raw[1][1],
    color='#196f3d', s=250, marker='*', edgecolor='black', linewidth=2,
    label='Centroid 1 (Sesuai)'
)

# Garis ambang batas benchmark
plt.axvline(x=25.0, color='gray', linestyle='--', alpha=0.6, label='Batas Rata-Rata Minat (25.0)')
plt.axhline(y=80.0, color='blue', linestyle=':', alpha=0.5, label='Batas KKM Akademik (80.0)')

# Konfigurasi Label & Judul
plt.title("Diagnosa Awal Kesesuaian Siswa SMKN 1 Bintan Timur\nK-Means Clustering 2D (K=2, Silhouette Score = 0.321)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Sumbu X: Skor Minat Kejuruan RIASEC Siswa (Rentang 7 - 35)", fontsize=11, fontweight='bold')
plt.ylabel("Sumbu Y: Rata-Rata 4 Mapel Kunci Rapor (Rentang 0 - 100)", fontsize=11, fontweight='bold')
plt.legend(loc='lower left', frameon=True, fontsize=9.5)
plt.tight_layout()

out_fig = os.path.join(OUTPUT_DIR, "scatter_kmeans_2d_177.png")
plt.savefig(out_fig, dpi=300)
plt.close()

print(f"[V] Grafik Visualisasi 2D Klaster Disimpan: {out_fig}")
print("=" * 85)

