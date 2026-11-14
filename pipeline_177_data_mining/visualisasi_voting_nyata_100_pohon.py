# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/visualisasi_voting_nyata_100_pohon.py
# Deskripsi: Visualisasi Empiris Hasil Voting Nyata 100 Pohon Keputusan Random Forest
#            (Bukan Ilustrasi Teori, Melainkan Komputasi Data Riil 36 Siswa Uji)
# ==============================================================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Memuat Model Random Forest Terlatih & Data Uji
path_model = os.path.join(OUTPUT_DIR, "model_random_forest_177.joblib")
path_data = os.path.join(BASE_DIR, "dataset_terklaster_177.csv")
path_pred = os.path.join(BASE_DIR, "hasil_prediksi_rf_177.csv")

rf_model = joblib.load(path_model)
df_all = pd.read_csv(path_data)
df_test = pd.read_csv(path_pred)

feature_cols = [
    'Matematika_scaled', 'Bahasa_Indonesia_scaled', 'Bahasa_Inggris_scaled', 'Produktif_scaled',
    'Realistic_scaled', 'Investigative_scaled', 'Artistic_scaled', 'Social_scaled', 'Enterprising_scaled', 'Conventional_scaled',
    'Klaster_KMeans'
]

# 2. Ekstraksi Suara Nyata dari 100 Pohon untuk Setiap Siswa di Data Uji
classes = list(rf_model.classes_)
n_trees = len(rf_model.estimators_)  # 100 pohon

voting_matrix = []

for idx, row in df_test.iterrows():
    sid = row['ID_Siswa']
    # Ambil fitur siswa tersebut
    s_feat = df_all[df_all['ID_Siswa'] == sid][feature_cols].values
    
    # Kumpulkan suara dari masing-masing 100 pohon keputusan
    votes = [estimator.predict(s_feat)[0] for estimator in rf_model.estimators_]
    
    # Hitung jumlah pohon yang memilih tiap kelas
    vote_dict = {cls_name: 0 for cls_name in classes}
    for v in votes:
        cls_name = classes[int(v)]
        vote_dict[cls_name] += 1
        
    voting_matrix.append(vote_dict)

df_votes = pd.DataFrame(voting_matrix)
df_votes['ID_Siswa'] = df_test['ID_Siswa'].values
df_votes['Prediksi'] = df_test['Prediksi_Random_Forest'].values
df_votes['Status_Prediksi'] = df_test['Status_Prediksi'].values
df_votes['Status_Kesesuaian'] = df_test['Status_Kesesuaian'].values

# Simpan rekapitulasi numerik voting 100 pohon ke CSV
path_votes_csv = os.path.join(OUTPUT_DIR, "rekapitulasi_voting_100_pohon_177.csv")
df_votes.to_csv(path_votes_csv, index=False)

# 3. Visualisasi Empiris Publikasi (Grafik Hasil Nyata Voting 100 Pohon)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), dpi=300, gridspec_kw={'width_ratios': [2.2, 1]})

# Warna resmi 5 Rumpun SNPMB
warna_rumpun = {
    'Ekonomi, Bisnis & Akuntansi': '#3498db',
    'Seni, Desain & Industri Kreatif': '#e74c3c',
    'Sains & Teknologi Informasi': '#2ecc71',
    'Pariwisata & Hospitality': '#f39c12',
    'Sosial, Humaniora & Pendidikan': '#9b59b6'
}

# --- SUBPLOT 1: Stacked Bar Chart Distribusi Suara 100 Pohon per Siswa Uji ---
y_indices = np.arange(len(df_test))
left_offset = np.zeros(len(df_test))

for cls_name in classes:
    counts = df_votes[cls_name].values
    ax1.barh(y_indices, counts, left=left_offset, height=0.75, 
             label=cls_name, color=warna_rumpun[cls_name], edgecolor='black', linewidth=0.3)
    left_offset += counts

# Label Y axis: ID Siswa dan Hasil Prediksinya
y_labels = [f"{row['ID_Siswa']} ({'Linier' if row['Status_Kesesuaian']=='Sesuai Jurusan' else 'Penyelamatan'})" 
            for _, row in df_test.iterrows()]
ax1.set_yticks(y_indices)
ax1.set_yticklabels(y_labels, fontsize=8)
ax1.set_xlabel("Akumulasi Jumlah Suara Pohon Keputusan (Total = 100 Suara Pohon)", fontsize=10, fontweight='bold')
ax1.set_ylabel("36 Siswa Data Uji Independen (Holdout Test Set)", fontsize=10, fontweight='bold')
ax1.set_xlim(0, 100)
ax1.set_title("HASIL KOMPUTASI NYATA MAJORITY VOTING DARI 100 POHON KEPUTUSAN\n(Distribusi Suara 100 Pohon Random Forest untuk Setiap Siswa Uji)", 
              fontsize=12, fontweight='bold', pad=12)
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True, fontsize=8.5)
ax1.axvline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Ambang Mayoritas (>50 Suara)')

# --- SUBPLOT 2: Distribusi Konsensus & Tingkat Keyakinan 100 Pohon ---
top_confidences = df_test['Confidence_Score (%)'].values
bins = [50, 60, 70, 80, 90, 100]
n_bins, _, patches_b = ax2.hist(top_confidences, bins=bins, color='#1abc9c', edgecolor='black', rwidth=0.85)

for p, count in zip(patches_b, n_bins):
    if count > 0:
        ax2.text(p.get_x() + p.get_width()/2, p.get_height() + 0.3, f"{int(count)} Siswa\n({count/len(df_test)*100:.1f}%)", 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel("Tingkat Keyakinan Konsensus Pemenang (% Suara dari 100 Pohon)", fontsize=10, fontweight='bold')
ax2.set_ylabel("Frekuensi Jumlah Siswa", fontsize=10, fontweight='bold')
ax2.set_ylim(0, max(n_bins) + 3)
ax2.set_title("DISTRIBUSI KEKUATAN KONSENSUS ENSEMBLE\n(Tingkat Kebulatan Suara 100 Pohon)", fontsize=12, fontweight='bold', pad=12)

# Box Insight Ilmiah
ax2.text(0.5, 0.40, 
         f"RINGKASAN DATA RIIL:\n"
         f"• Total Pohon: 100 Estimators\n"
         f"• Rata-Rata Konsensus: {top_confidences.mean():.2f}%\n"
         f"• Konsensus Tertinggi: {top_confidences.max():.2f}%\n"
         f"• Siswa dgn Konsensus ≥70%: {(top_confidences >= 70).sum()} / 36 siswa ({((top_confidences >= 70).mean()*100):.1f}%)\n"
         f"• Akurasi Prediksi Uji: 86.11%",
         transform=ax2.transAxes, ha='center', va='center', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.6", facecolor='#fef9e7', edgecolor='#f39c12', lw=1.5))

plt.tight_layout()
out_fig = os.path.join(OUTPUT_DIR, "hasil_voting_nyata_100_pohon_177.png")
plt.savefig(out_fig, dpi=300)
plt.close()

print("=" * 85)
print("[V] VISUALISASI HASIL VOTING NYATA 100 POHON BERHASIL DIBUAT!")
print(f"    - File Grafik Disimpan : {out_fig}")
print(f"    - File Tabel Suara CSV : {path_votes_csv}")
print("=" * 85)

