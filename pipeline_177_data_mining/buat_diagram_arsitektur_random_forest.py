# ==============================================================================
# PROYEK DATA MINING NALARA - SMKN 1 BINTAN TIMUR
# File: pipeline_177_data_mining/buat_diagram_arsitektur_random_forest.py
# Deskripsi: Menghasilkan Diagram Skema Arsitektur Kerja Random Forest Classifier
#            (Bootstrap Aggregating, Multi-Tree, dan Majority Voting)
# ==============================================================================

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inisialisasi Canvas Gambar Resolusi Tinggi (300 DPI)
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Warna Palet
c_input = '#2c3e50'
c_tree_bg = '#f8f9fa'
c_node_root = '#2980b9'
c_node_int = '#3498db'
c_leaf = '#27ae60'
c_vote = '#8e44ad'
c_output = '#d35400'

# 1. Judul Header Diagram
plt.text(50, 96, "ARSITEKTUR & MEKANISME KERJA RANDOM FOREST CLASSIFIER (NALARA)", 
         ha='center', va='center', fontsize=14, fontweight='bold', color='#1a252f')
plt.text(50, 93, "Ensemble Bagging dari 100 Decision Trees dengan Majority Voting untuk Rekomendasi Rumpun Prodi PT", 
         ha='center', va='center', fontsize=10, style='italic', color='#555555')

# 2. Box Data Masukan Siswa (Top Layer)
box_input = patches.FancyBboxPatch(
    (22, 80), 56, 9,
    boxstyle="round,pad=0.8,rounding_size=1.5",
    facecolor='#ebf5fb', edgecolor=c_input, linewidth=2
)
ax.add_patch(box_input)
plt.text(50, 85.5, "DATA INPUT SISWA (11 FITUR TERSTANDARISASI Z-SCORE)", 
         ha='center', va='center', fontsize=11, fontweight='bold', color=c_input)
plt.text(50, 82.2, "4 Rapor (Mat, Indo, Ing, Prod) + 6 Minat RIASEC (R, I, A, S, E, C) + 1 Diagnosa Kesesuaian K-Means", 
         ha='center', va='center', fontsize=9, color='#333333')

# 3. Posisi Horizontal 4 Pohon (Tree 1, Tree 2, Tree 3, Tree 100)
tree_x_centers = [15, 38, 62, 85]
tree_titles = ["Tree #1", "Tree #2", "Tree #3", "Tree #100"]
tree_results = [
    "Sains & IT",
    "Sains & IT",
    "Bisnis & Akuntansi",
    "Sains & IT"
]

# Panah dari Data Input ke Masing-Masing Pohon (Bootstrap Sampling)
for x in tree_x_centers:
    ax.annotate(
        "", xy=(x, 68), xytext=(50, 80),
        arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=1.8, ls="--")
    )
plt.text(50, 75.5, "Bootstrap Sampling (Data Acak dengan Pengembalian) & Seleksi Acak Fitur (m ≈ √11 = 3)", 
         ha='center', va='center', fontsize=8.5, style='italic', color='#444444', 
         bbox=dict(boxstyle="round,pad=0.3", fc="#fdfefe", ec="#bdc3c7", lw=1))

# 4. Menggambar Pohon-Pohon Keputusan (Decision Trees)
def draw_mini_tree(x_c, y_c, ax, node_color=c_node_root):
    # Root Node
    ax.add_patch(patches.Circle((x_c, y_c), 1.6, facecolor=node_color, edgecolor='black', lw=1))
    
    # Level 1 Nodes
    x_l1, y_l1 = x_c - 4.5, y_c - 5
    x_r1, y_r1 = x_c + 4.5, y_c - 5
    ax.plot([x_c, x_l1], [y_c, y_l1], color='#333333', lw=1.2)
    ax.plot([x_c, x_r1], [y_c, y_r1], color='#333333', lw=1.2)
    ax.add_patch(patches.Circle((x_l1, y_l1), 1.4, facecolor=c_node_int, edgecolor='black', lw=1))
    ax.add_patch(patches.Circle((x_r1, y_r1), 1.4, facecolor=c_node_int, edgecolor='black', lw=1))
    
    # Level 2 Nodes (Daun / Leaves)
    leaves = [
        (x_l1 - 2.5, y_l1 - 5), (x_l1 + 2.5, y_l1 - 5),
        (x_r1 - 2.5, y_r1 - 5), (x_r1 + 2.5, y_r1 - 5)
    ]
    for lx, ly in leaves[:2]:
        ax.plot([x_l1, lx], [y_l1, ly], color='#555555', lw=1)
        ax.add_patch(patches.Circle((lx, ly), 1.2, facecolor=c_leaf, edgecolor='black', lw=0.8))
    for rx, ry in leaves[2:]:
        ax.plot([x_r1, rx], [y_r1, ry], color='#555555', lw=1)
        ax.add_patch(patches.Circle((rx, ry), 1.2, facecolor=c_leaf, edgecolor='black', lw=0.8))

for i, x_pos in enumerate(tree_x_centers):
    # Background Box untuk Pohon
    bg = patches.FancyBboxPatch(
        (x_pos - 8.5, 41), 17, 26,
        boxstyle="round,pad=0.5,rounding_size=1",
        facecolor=c_tree_bg, edgecolor='#bdc3c7', lw=1.2
    )
    ax.add_patch(bg)
    plt.text(x_pos, 64.5, tree_titles[i], ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50')
    
    # Gambar struktur cabang
    draw_mini_tree(x_pos, 59, ax)
    
    # Hasil Prediksi Individual Pohon
    box_res = patches.FancyBboxPatch(
        (x_pos - 7.5, 42.5), 15, 4.5,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor='#e8f8f5', edgecolor=c_leaf, lw=1.2
    )
    ax.add_patch(box_res)
    plt.text(x_pos, 45.8, f"Prediksi #{i+1 if i<3 else 100}:", ha='center', va='center', fontsize=7.5, color='#555555')
    plt.text(x_pos, 44.0, tree_results[i], ha='center', va='center', fontsize=8.5, fontweight='bold', color='#16a085')

# Tanda Titik-Titik Horizontal (Mewakili Tree 4 s/d Tree 99)
plt.text(73.5, 54, ". . .", ha='center', va='center', fontsize=22, fontweight='bold', color='#7f8c8d')
plt.text(73.5, 50, "(Total 100 Pohon)", ha='center', va='center', fontsize=8.5, style='italic', color='#7f8c8d')

# 5. Lapisan Majority Voting (Agregasi Suara Mayoritas)
for x in tree_x_centers:
    ax.annotate(
        "", xy=(50, 27), xytext=(x, 41),
        arrowprops=dict(arrowstyle="->", color=c_vote, lw=1.5)
    )

box_vote = patches.FancyBboxPatch(
    (22, 17.5), 56, 10,
    boxstyle="round,pad=0.8,rounding_size=1.5",
    facecolor='#f4ecf7', edgecolor=c_vote, linewidth=2
)
ax.add_patch(box_vote)
plt.text(50, 24.8, "MEKANISME MAJORITY VOTING (AGREGASI 100 POHON)", 
         ha='center', va='center', fontsize=11, fontweight='bold', color=c_vote)
plt.text(50, 22.0, r"$\hat{Y} = \arg\max_c \sum_{b=1}^{100} I(h_b(x) = c) \quad \mid \quad \text{Confidence} = \frac{\text{Suara Pemenang}}{100} \times 100\%$", 
         ha='center', va='center', fontsize=9.5, color='#2c3e50')
plt.text(50, 19.3, "Rekap Siswa_018: Sains & IT = 74 suara | Bisnis = 19 | Seni = 6 | Pariwisata = 1 | Sosial = 0",
         ha='center', va='center', fontsize=8.5, fontweight='bold', color='#512e5f')

# 6. Lapisan Output Akhir (Rekomendasi Rumpun Prodi PT)
ax.annotate(
    "", xy=(50, 11), xytext=(50, 17.5),
    arrowprops=dict(arrowstyle="->", color=c_output, lw=2.5)
)

box_final = patches.FancyBboxPatch(
    (18, 2.5), 64, 8,
    boxstyle="round,pad=0.8,rounding_size=1.5",
    facecolor='#fef5e7', edgecolor=c_output, linewidth=2.5
)
ax.add_patch(box_final)
plt.text(50, 7.8, "OUTPUT REKOMENDASI RUMPUN PROGRAM STUDI PT", 
         ha='center', va='center', fontsize=11, fontweight='bold', color=c_output)
plt.text(50, 4.5, "Siswa_018: Sains & Teknologi Informasi (Keyakinan: 74.00%) | Status K-Means = 1 (Sesuai) | Tipe: Rekomendasi Linier", 
         ha='center', va='center', fontsize=8.5, fontweight='bold', color='#7e5109')

plt.tight_layout()

# Simpan Gambar Diagram Arsitektur
out_path = os.path.join(OUTPUT_DIR, "diagram_arsitektur_random_forest.png")
plt.savefig(out_path, dpi=300)
plt.close()

print("=" * 80)
print(f"[V] DIAGRAM ARSITEKTUR RANDOM FOREST BERHASIL DIBUAT DENGAN KODE!")
print(f"    - Lokasi File : {out_path}")
print(f"    - Resolusi    : 300 DPI (Standar Publikasi Ilmiah & Dokumen Skripsi)")
print("=" * 80)

