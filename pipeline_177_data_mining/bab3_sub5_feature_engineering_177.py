# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab3_sub5_feature_engineering_177.py
# Sub-Bab III.5: Feature Engineering & Triangulasi Target Label
# ==============================================================================

import os
import pandas as pd

print("=" * 85)
print("SUB-BAB III.5: FEATURE ENGINEERING & TRIANGULASI TARGET LABEL")
print("=" * 85)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Memuat Dataset Anonim dari Sub-Bab III.4
path_anonim = os.path.join(BASE_DIR, "dataset_anonim_177.csv")
df = pd.read_csv(path_anonim)

# 2. III.5.1 Pembentukan Fitur Sumbu Y K-Means: Rata-Rata Rapor 4 Mapel Kunci
# Formula: (Matematika + B. Indo + B. Ing + Produktif) / 4
df['Rata_Rapor'] = (
    df['Matematika'] + 
    df['Bahasa_Indonesia'] + 
    df['Bahasa_Inggris'] + 
    df['Produktif']
) / 4.0

# 3. III.5.2 Pembentukan Fitur Sumbu X K-Means: Skor Minat Kejuruan Spesifik
# Pemetaan: RPL -> Investigative, AKL -> Conventional, PH -> Social, DKV & TB -> Artistic
def get_skor_minat_kejuruan(row):
    jur = str(row['Jurusan']).upper()
    if 'REKAYASA' in jur or 'RPL' in jur:
        return row['Investigative']
    elif 'AKUNTANSI' in jur or 'AKT' in jur:
        return row['Conventional']
    elif 'PERHOTELAN' in jur or 'PH' in jur:
        return row['Social']
    elif 'DESAIN' in jur or 'DKV' in jur:
        return row['Artistic']
    elif 'BUSANA' in jur or 'TB' in jur:
        return row['Artistic']
    return row['Realistic']

df['Skor_Minat_Kejuruan'] = df.apply(get_skor_minat_kejuruan, axis=1)

# 4. III.5.3 Triangulasi Target Y Rekomendasi Program Studi PT (5 Rumpun SNPMB)
# Menggabungkan Pilar 1 (Pilihan Siswa), Pilar 2 (RIASEC), dan Pilar 3 (Rapor/Jurusan)
def triangulate_target_prodi(row):
    ans = str(row['Pilihan_Kuliah_Mentah']).strip()
    r = row['Realistic']; i = row['Investigative']; a = row['Artistic']
    s = row['Social']; e = row['Enterprising']; c = row['Conventional']
    jur = str(row['Jurusan']).upper()
    
    # Aturan 1: Siswa DKV (Koreksi opsi kuesioner yang tidak memuat pilihan Seni)
    if 'DKV' in jur:
        if a >= 24 or 'belum tahu' in ans.lower():
            return 'Seni, Desain & Industri Kreatif'
        elif 'ekonomi' in ans.lower() or 'bisnis' in ans.lower():
            return 'Ekonomi, Bisnis & Akuntansi'
        else:
            return 'Sains & Teknologi Informasi'
            
    # Aturan 2: Siswa Tata Busana (Koreksi opsi kuesioner)
    if 'BUSANA' in jur:
        if a >= 24 or 'belum tahu' in ans.lower():
            return 'Seni, Desain & Industri Kreatif'
        elif 'pariwisata' in ans.lower() or 'kuliner' in ans.lower():
            return 'Pariwisata & Hospitality'
        else:
            return 'Ekonomi, Bisnis & Akuntansi'
            
    # Aturan 3: Penyelamatan siswa bingung / belum tahu pada jurusan lain (Berdasarkan Dimensi Holland Tertinggi)
    if 'belum tahu' in ans.lower():
        scores = {'R': r, 'I': i, 'A': a, 'S': s, 'E': e, 'C': c}
        top_dim = max(scores, key=scores.get)
        if top_dim in ['I', 'R']:
            return 'Sains & Teknologi Informasi'
        elif top_dim in ['C', 'E']:
            return 'Ekonomi, Bisnis & Akuntansi'
        elif top_dim == 'A':
            return 'Seni, Desain & Industri Kreatif'
        else:
            return 'Pariwisata & Hospitality'
        
    # Aturan 4: Pemetaan pilihan eksplisit yang tervalidasi
    if 'teknologi' in ans.lower() or 'rekayasa' in ans.lower():
        return 'Sains & Teknologi Informasi'
    elif 'pariwisata' in ans.lower() or 'perhotelan' in ans.lower():
        return 'Pariwisata & Hospitality'
    elif 'ekonomi' in ans.lower() or 'akuntansi' in ans.lower() or 'bisnis' in ans.lower():
        return 'Ekonomi, Bisnis & Akuntansi'
    elif 'hukum' in ans.lower() or 'sosial' in ans.lower() or 'pendidikan' in ans.lower() or 'psikologi' in ans.lower():
        return 'Sosial, Humaniora & Pendidikan'
    else:
        return 'Sains & Teknologi Informasi'

df['Target_Rumpun_Kuliah'] = df.apply(triangulate_target_prodi, axis=1)

# 5. Menyimpan Dataset Ter-engineer
out_path = os.path.join(BASE_DIR, "dataset_engineered_177.csv")
df.to_csv(out_path, index=False)

print(f"\n[V] Hasil Sub-Bab III.5 Berhasil Disimpan: {os.path.basename(out_path)}")
print(f"    - Statistik Sumbu Y (Rata_Rapor)       : Min={df['Rata_Rapor'].min():.2f}, Max={df['Rata_Rapor'].max():.2f}, Mean={df['Rata_Rapor'].mean():.2f}")
print(f"    - Statistik Sumbu X (Skor_Minat_Kejuruan): Min={df['Skor_Minat_Kejuruan'].min():.2f}, Max={df['Skor_Minat_Kejuruan'].max():.2f}, Mean={df['Skor_Minat_Kejuruan'].mean():.2f}")
print("\n    - Distribusi Hasil Triangulasi Target Rekomendasi Kuliah (Target Y):")
for rmp, jml in df['Target_Rumpun_Kuliah'].value_counts().items():
    pct = (jml / len(df)) * 100
    print(f"      * {rmp:<33}: {jml:>2} siswa ({pct:.1f}%)")
print("=" * 85)

