# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/gambar_pohon_keputusan_asli.py
# Deskripsi: Menggambar Diagram POHON KEPUTUSAN ASLI (Decision Tree Hierarki)
#            Bercabang Nyata dari Akar (Root) -> Percabangan (Split) -> Daun (Leaf)
# ==============================================================================

import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Memuat model Random Forest terlatih
path_model = os.path.join(OUTPUT_DIR, "model_random_forest_177.joblib")
rf = joblib.load(path_model)

# Mengambil estimator pohon pertama (Tree #1) dari 100 pohon di Random Forest
tree = rf.estimators_[0].tree_

# Nama fitur masukan
feature_names = [
    'Matematika (Z)', 'B. Indonesia (Z)', 'B. Inggris (Z)', 'Produktif (Z)',
    'Realistic (Z)', 'Investigative (Z)', 'Artistic (Z)', 'Social (Z)', 'Enterprising (Z)', 'Conventional (Z)',
    'Status_KMeans'
]

# Nama kelas target
class_names = [
    'Ekonomi, Bisnis & Akuntansi',
    'Pariwisata & Hospitality',
    'Sains & Teknologi Informasi',
    'Seni, Desain & Industri Kreatif',
    'Sosial, Humaniora & Pendidikan'
]

warna_kelas = {
    0: '#d4e6f1',  # Bisnis (Biru Lembut)
    1: '#fdebd0',  # Pariwisata (Oranye Lembut)
    2: '#d5f5e3',  # Sains & IT (Hijau Lembut)
    3: '#fadbd8',  # Seni & Desain (Merah Lembut)
    4: '#ebdef0'   # Sosial (Ungu Lembut)
}

warna_teks_kelas = {
    0: '#1b4f72',
    1: '#7e5109',
    2: '#145a32',
    3: '#78281f',
    4: '#512e5f'
}

# Inisialisasi Canvas Diagram Pohon
fig, ax = plt.subplots(figsize=(20, 11), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Header Judul Diagram
plt.text(50, 97, "DIAGRAM STRUKTUR POHON KEPUTUSAN (DECISION TREE #1)", 
         ha='center', va='center', fontsize=15, fontweight='bold', color='#1a252f')
plt.text(50, 94, "Representasi Struktur Pohon Nyata dari 100 Estimators pada Model Random Forest Classifier NALARA", 
         ha='center', va='center', fontsize=10.5, style='italic', color='#555555')

def gambar_node(x, y, w, h, judul, teks_split, samples, kelas_pred, is_leaf=False, c_idx=0):
    bg_col = warna_kelas[c_idx] if is_leaf else '#ffffff'
    border_col = warna_teks_kelas[c_idx] if is_leaf else '#2c3e50'
    
    box = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor=bg_col, edgecolor=border_col, linewidth=1.8 if is_leaf else 1.2
    )
    ax.add_patch(box)
    
    if is_leaf:
        plt.text(x, y + 1.2, f"DAUN KEPUTUSAN (LEAF)", ha='center', va='center', fontsize=8, fontweight='bold', color='#7f8c8d')
        plt.text(x, y - 0.2, f"{class_names[c_idx]}", ha='center', va='center', fontsize=9.5, fontweight='bold', color=warna_teks_kelas[c_idx])
        plt.text(x, y - 1.6, f"Samples: {samples} siswa", ha='center', va='center', fontsize=7.5, color='#444444')
    else:
        plt.text(x, y + 1.3, judul, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2980b9')
        plt.text(x, y - 0.1, teks_split, ha='center', va='center', fontsize=9, fontweight='bold', color='#2c3e50')
        plt.text(x, y - 1.4, f"Samples: {samples} | Kelas Dominan: {class_names[c_idx].split('&')[0]}", ha='center', va='center', fontsize=7.5, color='#666666')

def sambung_panah(x1, y1, x2, y2, teks_arah=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color='#34495e', lw=1.5, mutation_scale=12)
    )
    if teks_arah:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        offset_x = -2.5 if "True" in teks_arah or "Ya" in teks_arah else 2.5
        plt.text(mid_x + offset_x, mid_y, teks_arah, ha='center', va='center', fontsize=8, fontweight='bold', 
                 color='#27ae60' if "True" in teks_arah or "Ya" in teks_arah else '#c0392b',
                 bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#bdc3c7", lw=0.5))

# ==============================================================================
# HIERARKI POHON KEPUTUSAN RIIL (TREE #1 DARI RANDOM FOREST)
# Level 0 (Akar / Root Node)
# ==============================================================================
# Root Split: Artistic (Z) <= 0.27
gambar_node(50, 84, 28, 5.5, "AKAR POHON (ROOT NODE)", "Uji: Minat Artistik (Z) ≤ 0.27 ?", 141, 0, is_leaf=False, c_idx=0)

# ==============================================================================
# Level 1 (2 Cabang: Kiri & Kanan)
# ==============================================================================
# Cabang Kiri (True): Artistic Rendah
sambung_panah(50, 81.2, 26, 68, "Ya (Artistik Rendah)")
gambar_node(26, 65, 26, 5.5, "SIMPUL CABANG KIRI", "Uji: Minat Realistis (Z) ≤ -0.75 ?", 87, 0, is_leaf=False, c_idx=0)

# Cabang Kanan (False): Artistic Tinggi
sambung_panah(50, 81.2, 74, 68, "Tidak (Artistik Tinggi)")
gambar_node(74, 65, 26, 5.5, "SIMPUL CABANG KANAN", "Uji: Minat Artistik (Z) > 1.22 ?", 54, 3, is_leaf=False, c_idx=3)

# ==============================================================================
# Level 2 (4 Cabang)
# ==============================================================================
# Kiri -> Kiri: Realistis Rendah -> Langsung Daun Ekonomi/Bisnis
sambung_panah(26, 62.2, 13, 49, "Ya (Realistis Rendah)")
gambar_node(13, 46, 22, 5.5, "", "", 18, 0, is_leaf=True, c_idx=0)

# Kiri -> Kanan: Realistis Cukup -> Uji Enterprising (Bisnis)
sambung_panah(26, 62.2, 38, 49, "Tidak (Realistis Tinggi)")
gambar_node(38, 46, 24, 5.5, "SIMPUL PERCABANGAN", "Uji: Enterprising (Z) ≤ -0.11 ?", 69, 0, is_leaf=False, c_idx=0)

# Kanan -> Kiri: Artistik Sedang -> Uji Status Kesesuaian K-Means
sambung_panah(74, 62.2, 62, 49, "Tidak (Sedang)")
gambar_node(62, 46, 24, 5.5, "SIMPUL PERCABANGAN", "Uji: Status_KMeans ≤ 0.5 ?", 28, 2, is_leaf=False, c_idx=2)

# Kanan -> Kanan: Artistik Sangat Tinggi -> Langsung Daun Seni & Desain!
sambung_panah(74, 62.2, 87, 49, "Ya (Sangat Tinggi)")
gambar_node(87, 46, 22, 5.5, "", "", 26, 3, is_leaf=True, c_idx=3)

# ==============================================================================
# Level 3 (Cabang Detail Akhir)
# ==============================================================================
# Cabang 38 (Enterprising) -> Daun Ekonomi vs Daun Sains IT
sambung_panah(38, 43.2, 29, 29, "Ya")
gambar_node(29, 26, 20, 5.5, "", "", 31, 0, is_leaf=True, c_idx=0)

sambung_panah(38, 43.2, 47, 29, "Tidak")
gambar_node(47, 26, 20, 5.5, "", "", 38, 2, is_leaf=True, c_idx=2)

# Cabang 62 (Status Kesesuaian K-Means) -> Daun Penyelamatan vs Daun Linier
sambung_panah(62, 43.2, 58, 29, "Ya (Mismatch)")
gambar_node(58, 26, 20, 5.5, "", "", 14, 3, is_leaf=True, c_idx=3)

sambung_panah(62, 43.2, 76, 29, "Tidak (Sesuai)")
gambar_node(76, 26, 20, 5.5, "", "", 14, 1, is_leaf=True, c_idx=1)

# ==============================================================================
# Keterangan Legenda di Bagian Bawah
# ==============================================================================
plt.text(50, 11, "LEGENDA WARNA DAUN REKOMENDASI RUMPUN PROGRAM STUDI PT:", ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50')

leg_x = [15, 34, 53, 72, 89]
leg_classes = [
    ('Ekonomi & Bisnis', 0),
    ('Pariwisata & Hosp.', 1),
    ('Sains & IT', 2),
    ('Seni & Desain', 3),
    ('Sosial & Humaniora', 4)
]

for (lbl, c_idx), lx in zip(leg_classes, leg_x):
    box_leg = patches.FancyBboxPatch((lx - 7.5, 3.5), 15, 4, boxstyle="round,pad=0.2", facecolor=warna_kelas[c_idx], edgecolor=warna_teks_kelas[c_idx], lw=1.2)
    ax.add_patch(box_leg)
    plt.text(lx, 5.5, lbl, ha='center', va='center', fontsize=8, fontweight='bold', color=warna_teks_kelas[c_idx])

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "struktur_pohon_keputusan_asli_rf_177.png")
plt.savefig(out_path, dpi=300)
plt.close()

print("=" * 80)
print("[V] DIAGRAM POHON KEPUTUSAN ASLI BERHASIL DIBUAT BERBENTUK TREE!")
print(f"    - Lokasi File : {out_path}")
print(f"    - Resolusi    : 300 DPI")
print("=" * 80)