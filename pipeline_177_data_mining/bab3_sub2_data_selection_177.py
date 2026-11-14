# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/bab3_sub2_data_selection_177.py
# Sub-Bab III.2: Data Selection (Seleksi 4 Mapel Kunci & Eliminasi Noise Rapor)
# ==============================================================================

import os
import glob
import pandas as pd
import numpy as np

print("=" * 80)
print("SUB-BAB III.2: DATA SELECTION (SELEKSI 4 MAPEL KUNCI DARI 29 LEGER RAPOR)")
print("=" * 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# 1. Mendeteksi Seluruh Buku Leger Rapor (Mengabaikan File Lock Temp ~$*)
pola_leger = os.path.join(ROOT_DIR, "Leger Semester Genap Tahun 2026/*/*.xlsx")
daftar_leger = sorted([f for f in glob.glob(pola_leger) if not os.path.basename(f).startswith("~$")])
print(f"[*] Ditemukan {len(daftar_leger)} file leger rapor valid.")

# 2. Kata Kunci Mata Pelajaran Kejuruan Produktif (5 Jurusan SMKN 1 Bintan Timur)
kata_kunci_produktif = [
    'REKAYASA', 'PERANGKAT LUNAK', 'AKUNTANSI', 'SPREADSHEET', 
    'PERHOTELAN', 'FOOD AND BEVERAGE', 'DESAIN KOMUNIKASI VISUAL', 
    'TATA BUSANA', 'BUSANA', 'BATIK', 'MEMBATIK', 'DASAR DASAR', 'DASAR-DASAR'
]

semua_rekor_rapor = []

# 3. Iterasi Pembacaan Tiap File Leger Rapor
for path_f in daftar_leger:
    nama_f = os.path.basename(path_f)
    df_raw = pd.read_excel(path_f, header=None)
    
    # Deteksi baris header dan kolom NIS
    b_header = None
    c_nis = None
    for r in range(6):
        row_str = [str(x).strip().upper() for x in df_raw.iloc[r].dropna().tolist()]
        if 'NIS' in row_str:
            b_header = r
            for c in range(len(df_raw.columns)):
                if str(df_raw.iloc[r, c]).strip().upper() == 'NIS':
                    c_nis = c
                    break
            break
            
    if b_header is None or c_nis is None:
        continue
        
    # Menentukan batas akhir kolom mata pelajaran (sebelum kolom presensi/kehadiran)
    c_batas = len(df_raw.columns)
    for c in range(len(df_raw.columns)):
        val_h = str(df_raw.iloc[b_header, c]).strip().upper()
        if 'KETIDAKHADIRAN' in val_h or 'SAKIT' in val_h:
            c_batas = c
            break
            
    # Ekstraksi nama mata pelajaran dari tabel legenda di bagian bawah leger
    daftar_mapel_text = []
    baris_ket_mapel = None
    for r in range(len(df_raw)):
        if 'KETERANGAN MAPEL' in str(df_raw.iloc[r].tolist()).upper():
            baris_ket_mapel = r
            break
            
    if baris_ket_mapel is not None:
        for r in range(baris_ket_mapel + 1, len(df_raw)):
            teks_row = [str(x).strip() for x in df_raw.iloc[r].dropna().tolist() if str(x).strip()]
            if teks_row:
                daftar_mapel_text.append(' '.join(teks_row))
                
    # III.2.1 Seleksi 4 Mapel Kunci & III.2.2 Eliminasi Mapel Non-Akademik / Noise
    col_mat = None
    col_bi = None
    col_en = None
    cols_prod = []
    
    for idx_m, nama_m in enumerate(daftar_mapel_text):
        c_aktual = (c_nis + 1) + idx_m
        if c_aktual >= c_batas:
            break
        m_upper = nama_m.upper()
        
        if 'MATEMATIKA' in m_upper:
            col_mat = c_aktual
        elif 'BAHASA INDONESIA' in m_upper and 'TINGKAT LANJUT' not in m_upper:
            col_bi = c_aktual
        elif 'BAHASA INGGRIS' in m_upper:
            col_en = c_aktual
        elif any(kunci in m_upper for kunci in kata_kunci_produktif):
            cols_prod.append(c_aktual)
            
    # Mengambil nilai numerik siswa per baris data
    for r in range(b_header + 4, len(df_raw)):
        no_str = str(df_raw.iloc[r, 0]).strip()
        if no_str.isdigit():
            nis_str = str(df_raw.iloc[r, c_nis]).strip().replace('.0', '')
            if nis_str and nis_str != 'nan':
                v_mat = float(df_raw.iloc[r, col_mat]) if col_mat and pd.notna(df_raw.iloc[r, col_mat]) else np.nan
                v_bi = float(df_raw.iloc[r, col_bi]) if col_bi and pd.notna(df_raw.iloc[r, col_bi]) else np.nan
                v_en = float(df_raw.iloc[r, col_en]) if col_en and pd.notna(df_raw.iloc[r, col_en]) else np.nan
                
                v_prod_vals = [float(df_raw.iloc[r, cp]) for cp in cols_prod if pd.notna(df_raw.iloc[r, cp])]
                v_prod = np.mean(v_prod_vals) if v_prod_vals else np.nan
                
                semua_rekor_rapor.append({
                    'NIS': nis_str,
                    'Matematika': v_mat,
                    'Bahasa_Indonesia': v_bi,
                    'Bahasa_Inggris': v_en,
                    'Produktif': v_prod
                })

# 4. Agregasi Rata-Rata Multi-Semester per NIS Siswa
df_rapor_agregat = pd.DataFrame(semua_rekor_rapor).groupby('NIS').mean().reset_index()

# 5. Menyimpan Hasil Seleksi Mapel ke File Sementara
out_path = os.path.join(BASE_DIR, "temp_rapor_selected_177.csv")
df_rapor_agregat.to_csv(out_path, index=False)

print(f"\n[V] Hasil Sub-Bab III.2 Berhasil Disimpan: {os.path.basename(out_path)}")
print(f"    - Total Rekor Mentah Terekstrak : {len(semua_rekor_rapor)} baris nilai")
print(f"    - Total Siswa Unik Rapor       : {len(df_rapor_agregat)} siswa")
print(f"    - 4 Mapel Kunci Terpilih       : Matematika, Bahasa Indonesia, Bahasa Inggris, Produktif")
print(f"    - Noise Mapel Tereliminasi     : Agama, PKn, PJOK, Seni Budaya, Sejarah, Ekstrakurikuler, Presensi")
print("=" * 80)

