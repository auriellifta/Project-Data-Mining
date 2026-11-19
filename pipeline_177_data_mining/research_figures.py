"""Gambar laporan dari estimator tersimpan; tidak melakukan fit atau mengubah data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

INK = '#203348'
MUTED = '#526476'
CLUSTER_COLORS = {0: '#d88327', 1: '#15977d'}
CLASS_COLORS = ['#3786b2', '#dc9639', '#269d78', '#d5646a', '#9064ad']
CLASS_NAMES = ['Bisnis', 'Pariwisata', 'Sains & IT', 'Seni & Desain', 'Sosial']
FEATURE_NAMES = ['Matematika Z', 'B. Indonesia Z', 'B. Inggris Z', 'Produktif Z',
                 'Realistic Z', 'Investigative Z', 'Artistic Z', 'Social Z',
                 'Enterprising Z', 'Conventional Z', 'Klaster K-Means']
FEATURE_SHORT = ['Mat Z', 'Indo Z', 'Ing Z', 'Prod Z', 'R Z', 'I Z', 'A Z',
                 'S Z', 'E Z', 'C Z', 'Klaster']


def _save(fig, output, name, *, vector=False, jpg=False):
    """PNG untuk Word, SVG untuk pembesaran diagram tanpa teks pecah."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', dpi=250, bbox_inches='tight', facecolor='white')
    if vector:
        with matplotlib.rc_context({'svg.fonttype': 'none'}):
            fig.savefig(output / f'{name}.svg', bbox_inches='tight', facecolor='white')
    if jpg:
        fig.savefig(output / f'{name}.jpg', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _style(ax):
    ax.set_facecolor('#fbfcfe')
    ax.grid(color='#dfe6ed', linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#c6d1dc')
    ax.tick_params(colors=MUTED)


def draw_kmeans(bundle, df, prep, km_features, output):
    centers = bundle['metadata']['centroids']
    fig, ax = plt.subplots(figsize=(11, 7.2))
    fig.subplots_adjust(left=.10, right=.97, top=.85, bottom=.17)
    _style(ax)
    for code, label in [(0, 'Tidak Sesuai'), (1, 'Sesuai')]:
        part = df[df.Klaster_KMeans == code]
        ax.scatter(part.Skor_Minat_Kejuruan, part.Rata_Rapor, s=62,
                   color=CLUSTER_COLORS[code], alpha=.75, edgecolors='white',
                   linewidths=.65, label=f'C{code} · {label} ({len(part)} siswa)')
    for c in centers:
        ax.scatter(c['x'], c['y'], marker='X', s=300, color=CLUSTER_COLORS[c['kode']],
                   edgecolor=INK, linewidth=1.4, zorder=5)
        ax.annotate(f"C{c['kode']}  ({c['x']:.2f}; {c['y']:.2f})",
                    (c['x'], c['y']), xytext=(-100, 23) if c['kode'] == 0 else (12, -36),
                    textcoords='offset points', fontsize=10, fontweight='bold', color=INK,
                    bbox={'boxstyle': 'round,pad=.4', 'fc': 'white', 'ec': '#cbd5df'},
                    arrowprops={'arrowstyle': '-', 'color': MUTED}, zorder=6)
    x = df.Skor_Minat_Kejuruan
    y = df.Rata_Rapor
    xpad = max(1.0, float(x.max() - x.min()) * .07)
    ypad = max(2.0, float(y.max() - y.min()) * .10)
    ax.set(xlim=(max(7, float(x.min()) - xpad), min(35, float(x.max()) + xpad)),
           ylim=(max(0, float(y.min()) - ypad), min(100, float(y.max()) + ypad)),
           xlabel='Skor minat kejuruan RIASEC', ylabel='Rata-rata empat mata pelajaran')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    fig.suptitle('Pengelompokan minat kejuruan dan nilai rapor', x=.5, ha='center',
                 y=.97, fontsize=19, fontweight='bold', color=INK)
    fig.text(.5, .91, 'K-Means 2D · K = 2 · 177 siswa',
             ha='center', fontsize=12, color=MUTED)
    ax.legend(loc='lower left', framealpha=.96, fontsize=10)
    fig.text(.1, .09, 'Tanda X: centroid dari 141 data latih. Tampilan diperbesar mengikuti rentang data; semua titik disertakan.',
             fontsize=9, color=MUTED)
    fig.text(.1, .055, 'Skala instrumen: X 7–35, Y 0–100. Jarak klaster dihitung dalam Z-score; label merupakan kategori operasional.',
             fontsize=9, color=MUTED)
    bounds = {'x': list(ax.get_xlim()), 'y': list(ax.get_ylim())}
    _save(fig, output, 'scatter_kmeans_2d_177', vector=True)

    sil = silhouette_samples(df[km_features], df.Klaster_KMeans)
    train = df[df.ID_Siswa.isin(prep['train_ids'])]
    train_sil = silhouette_samples(train[km_features], train.Klaster_KMeans)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    fig.subplots_adjust(left=.12, right=.96, top=.83, bottom=.20)
    _style(ax)
    lower = 8
    for code, label in [(0, 'Tidak Sesuai'), (1, 'Sesuai')]:
        values = np.sort(sil[df.Klaster_KMeans == code])
        positions = np.arange(lower, lower + len(values))
        ax.fill_betweenx(positions, 0, values, color=CLUSTER_COLORS[code], alpha=.8)
        ax.text(.99, lower + len(values) / 2, f'C{code} · {label}\n{len(values)} siswa',
                ha='right', va='center', color=INK, fontsize=11,
                transform=ax.get_yaxis_transform())
        lower += len(values) + 12
    ax.axvline(0, color=MUTED, lw=.8)
    ax.axvline(sil.mean(), color='#b33e52', linestyle='--', lw=2,
               label=f'Rata-rata 177 siswa: {sil.mean():.4f}')
    ax.set(xlim=(min(-.10, float(sil.min()) - .04), 1), ylim=(0, lower),
           xlabel='Koefisien silhouette', ylabel='Siswa, diurutkan di dalam setiap klaster', yticks=[])
    ax.legend(loc='lower right', framealpha=.96, fontsize=10)
    fig.suptitle('Evaluasi silhouette K-Means', x=.5, ha='center', y=.97,
                 fontsize=19, fontweight='bold', color=INK)
    fig.text(.5, .90, f'K = 2  |  Seluruh siswa: {sil.mean():.4f}  |  Data latih: {train_sil.mean():.4f}',
             ha='center', fontsize=12, color=MUTED)
    fig.text(.12, .105, 'Area berwarna mencakup 177 siswa. Model dan centroid dibentuk hanya dari 141 data latih.',
             fontsize=9, color=MUTED)
    fig.text(.12, .065, 'Nilai mendekati 1: kelompok lebih terpisah; mendekati 0: berdekatan; negatif: lebih dekat ke kelompok lain.',
             fontsize=9, color=MUTED)
    _save(fig, output, 'evaluasi_silhouette_kmeans_177', vector=True)
    return {'scatter_bounds': bounds, 'silhouette_all': float(sil.mean()),
            'silhouette_train': float(train_sil.mean()), 'scatter_rows': len(df)}


def _positions(tree, max_depth=None):
    """Susunan daun tetap mengikuti cabang asli estimator; tidak menyederhanakan aturan."""
    nodes, edges = {}, []
    next_x = 0

    def visit(node, depth):
        nonlocal next_x
        leaf = tree.children_left[node] == -1
        stopped = max_depth is not None and depth >= max_depth and not leaf
        if leaf or stopped:
            x = next_x
            next_x += 1
        else:
            left, right = int(tree.children_left[node]), int(tree.children_right[node])
            lx = visit(left, depth + 1)
            rx = visit(right, depth + 1)
            x = (lx + rx) / 2
            edges.extend([(node, left, True), (node, right, False)])
        nodes[node] = {'x': x, 'depth': depth, 'leaf': leaf, 'stopped': stopped}
        return x

    visit(0, 0)
    return nodes, edges, max(next_x - 1, 1)


def _node_table(estimator):
    tree = estimator.tree_
    rows = []
    for node in range(tree.node_count):
        leaf = tree.children_left[node] == -1
        weights = np.asarray(tree.value[node][0], dtype=float)
        # sklearn dapat menyimpan value sebagai proporsi atau jumlah berbobot.
        weights = weights / weights.sum() * tree.weighted_n_node_samples[node]
        rows.append({'Node': node, 'Jenis': 'Daun' if leaf else 'Percabangan',
                     'Fitur': '' if leaf else FEATURE_NAMES[tree.feature[node]],
                     'Ambang_Z_atau_biner': '' if leaf else float(tree.threshold[node]),
                     'Anak_kiri_jika_le': int(tree.children_left[node]),
                     'Anak_kanan_jika_gt': int(tree.children_right[node]),
                     'Sampel_unik': int(tree.n_node_samples[node]),
                     'Jumlah_berbobot': float(tree.weighted_n_node_samples[node]),
                     'Gini': float(tree.impurity[node]),
                     'Kelas_dominan': CLASS_NAMES[int(weights.argmax())],
                     'Bobot_per_kelas': json.dumps(weights.round(8).tolist())})
    return pd.DataFrame(rows)


def _tree_legend(fig, y=.055, fontsize=10):
    handles = [Line2D([], [], marker='o', linestyle='', markersize=9, color=color,
                      label=name) for name, color in zip(CLASS_NAMES, CLASS_COLORS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, y),
               ncol=5, frameon=False, fontsize=fontsize)


def draw_actual_tree(bundle, output):
    estimator = bundle['rf_model'].estimators_[0]
    tree = estimator.tree_
    _node_table(estimator).to_csv(Path(output) / 'detail_node_tree_1_177.csv', index=False)
    for name, depth in [('struktur_pohon_keputusan_asli_rf_177', 3),
                        ('visualisasi_pohon_keputusan_rf_177', None)]:
        nodes, edges, width = _positions(tree, depth)
        full = depth is None
        fig, ax = plt.subplots(figsize=(18, 12) if full else (17, 9.5))
        fig.subplots_adjust(left=.025, right=.975, top=.86 if full else .82, bottom=.21)
        ax.axis('off')
        step = 2.25
        for parent, child, left in edges:
            a, b = nodes[parent], nodes[child]
            x1, y1 = a['x'] * step, -a['depth'] * 2.4
            x2, y2 = b['x'] * step, -b['depth'] * 2.4
            ax.annotate('', xy=(x2, y2 + .30), xytext=(x1, y1 - 1.0),
                        arrowprops={'arrowstyle': '-|>', 'lw': 1.2, 'color': '#91a2b2'})
            if not full or a['depth'] < 2:
                ax.text((x1 + x2) / 2, (y1 + y2) / 2 - .20,
                        'Ya (≤)' if left else 'Tidak (>)', fontsize=9, color=MUTED,
                        ha='center', bbox={'fc': 'white', 'ec': 'none', 'pad': 1})
        for node, item in nodes.items():
            x, y = item['x'] * step, -item['depth'] * 2.4
            klass = int(tree.value[node][0].argmax())
            ax.scatter([x], [y], s=600 if full else 900, color=CLASS_COLORS[klass],
                       edgecolor=INK, linewidth=1.2, zorder=3)
            ax.text(x, y, str(node), ha='center', va='center', color='white',
                    fontsize=12, fontweight='bold', zorder=4)
            if item['leaf']:
                label = ('Daun\n' + ['Bisnis', 'Wisata', 'TI', 'Seni', 'Sosial'][klass]
                         if full else f'Daun · {CLASS_NAMES[klass]}')
            else:
                feature=(FEATURE_SHORT if full else FEATURE_NAMES)[tree.feature[node]]
                label = f'{feature}\n≤ {tree.threshold[node]:.3f}'
            ax.text(x, y - .48, label, ha='center', va='top', fontsize=12 if full else 10,
                    color=INK, linespacing=1.25,
                    bbox={'fc': 'white', 'ec': 'none', 'pad': .5})
            if item['stopped']:
                ax.text(x, y - 1.24, '↓ cabang berlanjut', ha='center', va='top',
                        fontsize=8, color=MUTED)
        max_level = max(item['depth'] for item in nodes.values())
        ax.set(xlim=(-1.65, width * step + 1.65), ylim=(-max_level * 2.4 - 1.75, .7))
        fig.suptitle('Struktur pohon keputusan #1', fontsize=23, fontweight='bold',
                     color=INK, x=.5, ha='center', y=.97)
        fig.text(.5, .91,
                 f"Seluruh {tree.node_count} node dan {estimator.get_n_leaves()} daun" if full else 'Cuplikan hingga kedalaman 3 · cabang lanjutan ditandai',
                 ha='center', fontsize=13, color=MUTED)
        fig.text(.04, .15,
                 'R: Realistic · I: Investigative · A: Artistic · S: Social · E: Enterprising · C: Conventional'
                 if full else 'Lingkaran: nomor node aktual. Aturan dan cabang dibaca dari estimator tersimpan; Z = fitur terstandardisasi.',
                 fontsize=10, color=MUTED)
        fig.text(.04, .12,
                 'Z: skor terstandardisasi · Mat/Indo/Ing/Prod: empat mapel · label daun mengikuti legenda warna'
                 if full else 'Warna: kelas dominan pada node. Keluaran pohon ini menjadi satu kontribusi dari 100 pohon RF.',
                 fontsize=10, color=MUTED)
        _tree_legend(fig, .055, fontsize=12 if full else 10)
        _save(fig, output, name, vector=True)


def _box(ax, xy, width, height, title, body='', *, color='#3678a0', fill='#edf5fb', title_size=12):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=.012,rounding_size=.014',
                               linewidth=1.5, facecolor=fill, edgecolor=color, zorder=2))
    ax.text(x + width / 2, y + height * (.66 if body else .5), title,
            fontsize=title_size, color=color, ha='center', va='center', fontweight='bold', zorder=3)
    if body:
        ax.text(x + width / 2, y + height * .28, body,
                fontsize=10, color=INK, ha='center', va='center', linespacing=1.4, zorder=3)


def _arrow(ax, start, end, color='#8fa0b1', dashed=False):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 1.5,
                            'linestyle': '--' if dashed else '-'}, zorder=1)


def _mini_tree(ax, estimator, sample, code, *, numbered=False):
    tree = estimator.tree_
    nodes, edges, width = _positions(tree, 2)
    path = set(estimator.decision_path(sample).indices)
    xy = {node: (.1 + .8 * item['x'] / width, .83 - item['depth'] * .22)
          for node, item in nodes.items()}
    for parent, child, _ in edges:
        chosen = parent in path and child in path
        ax.plot([xy[parent][0], xy[child][0]], [xy[parent][1], xy[child][1]],
                color='#e4872f' if chosen else '#c6d0da', lw=2.5 if chosen else 1.0, zorder=1)
    for node, item in nodes.items():
        x, y = xy[node]
        klass = int(tree.value[node][0].argmax())
        chosen = node in path
        ax.scatter([x], [y], s=420 if numbered else 200, color=CLASS_COLORS[klass],
                   edgecolor='#e4872f' if chosen else 'white', linewidth=2 if chosen else 1, zorder=3)
        if numbered:
            ax.text(x, y, str(node), ha='center', va='center', color='white',
                    fontsize=8, fontweight='bold', zorder=4)
        label = CLASS_NAMES[klass] if item['leaf'] else f'{FEATURE_SHORT[tree.feature[node]]} ≤ {tree.threshold[node]:.2f}'
        if item['stopped']:
            label += '\n…'
        ax.text(x, y - .06, label, fontsize=7.5, ha='center', va='top', color=INK)
    predicted = int(estimator.predict(sample)[0])
    ax.text(.5, .98, f'Pohon #{code}', ha='center', va='top', fontsize=13,
            fontweight='bold', color=INK)
    ax.text(.5, .08, f'Suara: {CLASS_NAMES[predicted]}', ha='center', va='center',
            fontsize=11, fontweight='bold', color=CLASS_COLORS[predicted],
            bbox={'boxstyle': 'round,pad=.4', 'fc': 'white', 'ec': CLASS_COLORS[predicted]})
    ax.set(xlim=(-.06, 1.06), ylim=(-.02, 1.03))
    ax.axis('off')


def draw_forest_architecture(bundle, df, rf_features, output):
    rf = bundle['rf_model']
    row = df[df.ID_Siswa == 'Siswa_018'].iloc[0] if (df.ID_Siswa == 'Siswa_018').any() else df.iloc[0]
    sample = row[rf_features].to_numpy(dtype=float).reshape(1, -1)
    per_tree = np.array([t.predict_proba(sample)[0] for t in rf.estimators_])
    votes = np.bincount(per_tree.argmax(axis=1), minlength=len(rf.classes_))
    probability = per_tree.mean(axis=0)
    predicted = int(probability.argmax())
    pure = all(np.all(t.tree_.impurity[t.tree_.children_left == -1] < 1e-10) for t in rf.estimators_)
    from inference_engine import NalaraInferenceEngine
    major = NalaraInferenceEngine.normalize_jurusan(str(row.Jurusan))
    aligned = {'RPL': 2, 'AKL': 0, 'PH': 1, 'DKV': 3, 'TB': 3}
    relation = 'Linier' if aligned[major] == predicted else 'Lintas bidang'

    for name, numbered in [('diagram_arsitektur_random_forest', False),
                            ('diagram_pohon_lingkaran_rf', True)]:
        fig = plt.figure(figsize=(18, 12.8))
        ax = fig.add_axes([.025, .015, .95, .97])
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        ax.text(.5, .975, 'Arsitektur Random Forest · 100 pohon', ha='center', va='center',
                fontsize=23, fontweight='bold', color=INK)
        ax.text(.5, .938, f'Contoh inferensi aktual {row.ID_Siswa}',
                ha='center', fontsize=12, color=MUTED)
        _box(ax, (.13, .825), .74, .075, 'INPUT SISWA · 11 FITUR',
             f'4 rapor Z-score + 6 skor RIASEC Z-score + kode K-Means {int(row.Klaster_KMeans)}')
        panel_lefts = [.01, .255, .5, .755]
        for left, tree_id in zip(panel_lefts, [0, 1, 2, 99]):
            _arrow(ax, (.5, .813), (left + .116, .758))
            ax.add_patch(FancyBboxPatch((left, .413), .232, .345,
                         boxstyle='round,pad=.004,rounding_size=.012', facecolor='#f7f9fc',
                         edgecolor='#d4dde7', linewidth=1.2))
            mini = ax.inset_axes([left + .006, .425, .220, .318])
            _mini_tree(mini, rf.estimators_[tree_id], sample, tree_id + 1, numbered=numbered)
            _arrow(ax, (left + .116, .407), (.5, .34), color='#8a62a9')
        ax.text(.5, .782, 'Siswa yang sama melewati seluruh pohon tersimpan; panel menampilkan cuplikan cabang asli.',
                ha='center', fontsize=10, color=MUTED,
                bbox={'fc': 'white', 'ec': 'none', 'pad': 2})
        ax.text(.5, .385, 'Pohon #4–#99 juga dihitung · garis oranye menunjukkan jalur contoh siswa',
                ha='center', fontsize=10, color=MUTED,
                bbox={'fc': 'white', 'ec': 'none', 'pad': 1})
        tally = '   |   '.join(f'{label}: {int(n)}' for label, n in zip(CLASS_NAMES, votes))
        _box(ax, (.10, .225), .80, .105, 'MAJORITY VOTING · AGREGASI 100 POHON',
             tally + '\nPrediksi implementasi: kelas dengan rata-rata probabilitas pohon tertinggi.',
             color='#8555a1', fill='#f4eef9', title_size=14)
        _arrow(ax, (.5, .213), (.5, .182), color='#cc792b')
        _box(ax, (.10, .085), .80, .085, f'FINAL CLASS · {rf.classes_[predicted]}',
             f'Confidence {probability[predicted]:.0%}  |  K-Means: {row.Status_Kesesuaian}  |  Hubungan: {relation}',
             color='#b66a24', fill='#fff5e8', title_size=14)
        ax.text(.5, .045, 'Pada bundle ini semua daun murni satu kelas: rata-rata probabilitas = jumlah suara / 100.'
                if pure else 'Suara mayoritas adalah rekap tambahan; kelas akhir mengikuti rata-rata probabilitas pohon.',
                ha='center', fontsize=10, color=MUTED)
        ax.text(.5, .016, 'Sampling data dengan pengembalian dan pemilihan fitur acak berlangsung saat pelatihan pada 141 siswa.',
                ha='center', fontsize=10, color=MUTED)
        _save(fig, output, name, vector=True, jpg=numbered)

    # Nama lama tetap disediakan sebagai alias berisi diagram model revisi.
    import shutil
    shutil.copy2(Path(output) / 'diagram_arsitektur_random_forest.png', Path(output) / 'diagram_pohon_final_rf.png')
    from PIL import Image
    with Image.open(Path(output) / 'diagram_pohon_final_rf.png') as image:
        image.convert('RGB').save(Path(output) / 'diagram_pohon_final_rf.jpg', quality=95)
    draw_hybrid_flow(output)
    return {'example_id': str(row.ID_Siswa), 'tree_ids_shown': [1, 2, 3, 100],
            'votes': dict(zip(rf.classes_.tolist(), votes.tolist())),
            'probabilities': dict(zip(rf.classes_.tolist(), probability.tolist())),
            'final_class': str(rf.classes_[predicted]), 'pure_leaves': pure,
            'major': major, 'relation': relation, 'cluster': int(row.Klaster_KMeans)}


def draw_hybrid_flow(output):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.text(.5, .96, 'Alur prediksi NALARA', ha='center', fontsize=21, fontweight='bold', color=INK)
    ax.text(.5, .91, 'Penerapan pada satu siswa', ha='center', fontsize=11, color=MUTED)
    blocks = [(.77, 'INPUT', '42 jawaban RIASEC + jurusan asal + empat nilai rapor'),
              (.60, 'PERSIAPAN FITUR', 'Enam skor komposit 7–35; transformasi memakai scaler tersimpan'),
              (.43, 'TAHAP 1 · K-MEANS 2D', 'X: minat terkait jurusan; Y: rerata rapor → kode kelompok 0 / 1'),
              (.26, 'TAHAP 2 · RANDOM FOREST', '10 fitur Z-score + 1 kode klaster → agregasi 100 pohon'),
              (.09, 'HASIL', 'Rumpun studi + confidence + contoh prodi; hubungan linier / lintas bidang')]
    for y, title, body in blocks:
        _box(ax, (.06, y), .88, .10, title, body)
    for a, b in zip(blocks, blocks[1:]):
        _arrow(ax, (.5, a[0] - .013), (.5, b[0] + .113))
    ax.text(.5, .018, 'Scaler dan kedua model dilatih pada 141 siswa. Prediksi baru tidak melatih ulang model.',
            ha='center', fontsize=10, color=MUTED)
    _save(fig, output, 'diagram_alur_hybrid_177', vector=True)
