# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab5_sub1_evaluasi_kmeans_177.py
# Sub-Bab V.1: Evaluasi Model Stage 1 - K-Means Clustering 2D
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

print("=" * 85)
print("SUB-BAB V.1: EVALUASI MODEL STAGE 1 - K-MEANS CLUSTERING 2D")
print("=" * 85)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Memuat Dataset Hasil Clustering Stage 1
path_data = os.path.join(BASE_DIR, "dataset_terklaster_177.csv")
df = pd.read_csv(path_data)

X_scaled = df[['Skor_Minat_Kejuruan_scaled', 'Rata_Rapor_scaled']].values
cluster_labels = df['Klaster_KMeans'].values
n_clusters = 2

# 2. Perhitungan Silhouette Score Keseluruhan & Per Klaster
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

sil_c0 = sample_silhouette_values[cluster_labels == 0]
sil_c1 = sample_silhouette_values[cluster_labels == 1]

print(f"[*] METRIK EVALUASI K-MEANS (STAGE 1):")
print(f"    - Silhouette Score Rata-Rata : {silhouette_avg:.4f} (Kategori Struktur Sedang / Cukup Kuat)")
print(f"    - Klaster 0 [Mismatch]       : Mean Silhouette = {sil_c0.mean():.4f} (Rentang: {sil_c0.min():.4f} s/d {sil_c0.max():.4f})")
print(f"    - Klaster 1 [Sesuai Jurusan] : Mean Silhouette = {sil_c1.mean():.4f} (Rentang: {sil_c1.min():.4f} s/d {sil_c1.max():.4f})")

# 3. Visualisasi Silhouette Plot (300 DPI)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

y_lower = 10
colors = ['#e67e22', '#27ae60']
cluster_names = ['Klaster 0: Tidak Sesuai (Mismatch)', 'Klaster 1: Sesuai Jurusan']

for i in range(n_clusters):
    ith_cluster_sil_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_sil_values.sort()
    
    size_cluster_i = ith_cluster_sil_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = colors[i]
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_sil_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.75,
        label=f"{cluster_names[i]} (n={size_cluster_i})"
    )
    y_lower = y_upper + 10

# Garis batas rata-rata Silhouette
ax1.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=1.5, label=f"Rata-Rata Silhouette ({silhouette_avg:.3f})")
ax1.set_title("Silhouette Plot Klaster K-Means 2D (K=2)\nEvaluasi Pemisahan Klaster Kesesuaian Siswa SMKN 1 Bintan Timur", fontsize=12, fontweight='bold', pad=15)
ax1.set_xlabel("Koefisien Nilai Silhouette (Rentang -1 s/d +1)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Indeks Siswa dalam Klaster", fontsize=10, fontweight='bold')
ax1.set_yticks([])
ax1.set_xlim([-0.15, 0.65])
# Menempatkan kotak legenda di posisi bawah (horizontal) agar tidak menghalangi diagram siluet sama sekali
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, fontsize=9.5, facecolor='#ffffff', edgecolor='#bdc3c7', framealpha=0.95)

out_fig = os.path.join(OUTPUT_DIR, "evaluasi_silhouette_kmeans_177.png")
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[V] Grafik Silhouette Plot Berhasil Disimpan : {out_fig}")
print("=" * 85)

