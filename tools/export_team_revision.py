"""Bangun paket tim versi 2 dari metrik/objek tersimpan, tanpa data identitas."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import shutil
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'pipeline_177_data_mining'
OUT=BASE/'output_model'
PACKAGE=ROOT/'Bahan_Laporan_Akhir_Untuk_Tim_v2'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def write_docx(path,text):
    """DOCX OpenXML mandiri; naskah editable, grafik disediakan di paket PNG."""
    body=[]
    lines=text.splitlines()
    cursor=0
    while cursor<len(lines):
        line=lines[cursor]
        cursor+=1
        stripped=line.strip()
        if not stripped: continue
        if stripped.startswith('|'):
            rows=[stripped]
            while cursor<len(lines) and lines[cursor].strip().startswith('|'):
                rows.append(lines[cursor].strip());cursor+=1
            xml_rows=[]
            for row in rows:
                cells=[v.strip() for v in row.strip('|').split('|')]
                if all(v and set(v)<=set('-: ') for v in cells):continue
                xml_cells=[]
                for cell in cells:
                    value=escape(cell.replace('**','').replace('`',''))
                    xml_cells.append('<w:tc><w:tcPr><w:tcW w:w="2000" w:type="dxa"/></w:tcPr><w:p><w:r><w:t>'+value+'</w:t></w:r></w:p></w:tc>')
                xml_rows.append('<w:tr>'+''.join(xml_cells)+'</w:tr>')
            border=''.join(f'<w:{edge} w:val="single" w:sz="4" w:color="B0B0B0"/>' for edge in ['top','left','bottom','right','insideH','insideV'])
            column_count=len(rows[0].strip('|').split('|'))
            grid='<w:tblGrid>'+''.join('<w:gridCol w:w="2000"/>' for _ in range(column_count))+'</w:tblGrid>'
            body.append('<w:tbl><w:tblPr><w:tblBorders>'+border+'</w:tblBorders></w:tblPr>'+grid+''.join(xml_rows)+'</w:tbl>')
            continue
        level=len(stripped)-len(stripped.lstrip('#')) if stripped.startswith('#') else 0
        style=f'Heading{min(level,3)}' if level else 'Normal'
        value=stripped.lstrip('#').strip() if level else stripped
        value=value.replace('**','').replace('`','')
        body.append(f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{escape(value)}</w:t></w:r></w:p>')
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    document=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{ns}"><w:body>'+''.join(body)+'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body></w:document>'
    styles=f'<w:styles xmlns:w="{ns}"><w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:style>'+''.join(f'<w:style w:type="paragraph" w:styleId="Heading{n}"><w:name w:val="heading {n}"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="{n-1}"/></w:pPr><w:rPr><w:b/><w:sz w:val="{34-4*n}"/></w:rPr></w:style>' for n in [1,2,3])+'</w:styles>'
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr('word/_rels/document.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        z.writestr('word/document.xml',document);z.writestr('word/styles.xml',styles)


def export():
    m=json.loads((OUT/'metrics_177.json').read_text(encoding='utf-8'))
    instrument=json.loads((BASE/'output_eda/instrument_metrics_177.json').read_text(encoding='utf-8'))
    km=m['kmeans']; centers=km['centroids']; c0,c1=centers
    source_map=table(['Subbab','Sumber utama'],[
        ['II','step1_data_understanding_177.py; instrument_validation.py'],
        ['III.1–III.5','bab3_sub1 ... bab3_sub5 (pembersihan, integrasi, pseudonim, fitur, target proxy)'],
        ['III.6','bab3_sub6_data_transformation_177.py → research_pipeline.transform_dataset'],
        ['IV.1','bab4_sub1_kmeans_177.py → research_pipeline.fit_kmeans'],
        ['IV.2','bab4_sub2_random_forest_177.py → research_pipeline.fit_forest'],
        ['V','bab5_sub1/bab5_sub2; research_pipeline.evaluate dan fungsi plot'],
        ['VI','inference_engine.py, api_server.py, app.py, demo_aplikasi_nalara.py']])
    per_class=[]
    for label in m['classes']:
        row=m['classification_report_all_labels'][label]; support=m['test_support'][label]
        per_class.append([label,support,f'{row["precision"]:.2%}' if support else 'Belum terukur',
                          f'{row["recall"]:.2%}' if support else 'Belum terukur',
                          f'{row["f1-score"]:.2%}' if support else 'Belum terukur'])
    text=f'''# RINGKASAN PROYEK DATA MINING NALARA — REVISI V2

Versi model: {m['model_version']} · Pipeline: {m['pipeline_version']} · 19 November 2026.
Sumber angka: output_model/metrics_177.json. Naskah ini menggantikan angka dan klaim draf sebelumnya.

## BAB I — Tujuan dan batas penelitian

NALARA mengeksplorasi profil minat kejuruan dan kemampuan akademik 177 siswa aktif SMKN 1 Bintan Timur, lalu memprediksi lima rumpun studi lanjut. Hasil digunakan sebagai bahan diskusi konseling. Data berasal dari siswa kelas X, XI, XII, bukan rekam longitudinal 177 siswa selama tiga tahun.

Rancangan dipertahankan: 42 butir, skor komposit enam dimensi 7–35, K-Means dua dimensi dengan K=2, dan Random Forest 100 pohon. Label Sesuai/Tidak Sesuai adalah definisi operasional kelompok relatif terhadap dua fitur; keanggotaan tidak mensyaratkan dua batas minimum dan tidak membuktikan seseorang salah jurusan.

## BAB II — Data understanding dan diagnostik instrumen

Sumber terdiri dari 29 leger semester genap 2026 dan satu kuesioner 177 responden. Audit integritas menemukan seluruh 177 NIS dan nama setelah normalisasi cocok; tidak ada jawaban 42 butir yang kosong/tidak dikenal. Ekstraksi mapel produktif mengikuti legenda mapel tiap leger; bobot setara mapel produktif perlu dikonfirmasi sebagai keputusan domain.

Komposisi jurusan: RPL 41, AKT/AKL 39, PH 38, DKV 37, TB 22. NIS/Nama hanya dipakai dalam proses integrasi privat; tabel model dan paket tim memakai ID pseudonim.

Alpha per dimensi:
{table(['Dimensi','Cronbach alpha'],[[name,f'{value:.4f}'] for name,value in instrument['alpha'].items()])}

Alpha Artistic yang rendah diungkapkan sebagai keterbatasan. Standardisasi tidak memperbaiki reliabilitas. Korelasi item-total memasukkan butirnya sendiri, sehingga disertakan item-rest yang mengeluarkan butir tersebut. Sebanyak {instrument['item_rest_flagged']} butir perlu ditelaah berdasar acuan korelasi item-rest eksploratif; 42 butir tetap digunakan pada instrumen versi ini demi konsistensi prototipe, dengan keterbatasan tersebut dinyatakan terbuka. Ini bukan kesimpulan bahwa seluruh instrumen valid secara isi/konstruk. Validasi pakar dan kajian instrumen lanjutan belum dibuktikan oleh perhitungan korelasi saja.

Korelasi nilai rapor–RIASEC disediakan beserta p-value presisi. Pengujian eksploratif belum dikoreksi untuk pengujian berganda dan tidak mengukur sebab-akibat. p yang kecil tidak otomatis berarti hubungan kuat; p-value tidak ditulis sebagai nol.

## BAB III — Preprocessing dan asal target

1. Bersihkan NIS, nama, jurusan dan petakan lima pilihan Likert resmi ke 1–5; hentikan proses jika identitas, jawaban, atau join tidak lengkap.
2. Pilih Matematika, Bahasa Indonesia, Bahasa Inggris, dan Produktif. Produktif merupakan rerata mapel produktif terpilih.
3. Inner join satu-ke-satu menggunakan NIS. Jangan menghapus siswa tanpa melaporkan kegagalan pencocokan.
4. Pseudonimisasi menjadi Siswa_001 ... Siswa_177 dan pisahkan file identitas dari paket distribusi.
5. Hitung skor komposit 5 butir Minat + 2 Kemampuan per dimensi. X adalah skor dimensi relevan jurusan: RPL→Investigative, AKL→Conventional, PH→Social, DKV/TB→Artistic. Y=(Matematika+Bahasa Indonesia+Bahasa Inggris+Produktif)/4.
6. Bentuk target proxy rumpun kuliah menggunakan aturan pilihan siswa, jurusan, dan skor RIASEC. Fungsi target ini tidak membaca rapor; rapor dipakai sebagai fitur prediksi. Jangan menyebutnya triangulasi tiga pilar atau ground truth penilaian BK independen.
7. Bagi 141 latih / 36 uji secara stratified, random_state=42, sebelum fit scaler atau model. StandardScaler fit pada 141 latih; transform yang sama dipakai untuk kedua bagian. Z-score menyetarakan pusat/skala, tidak menjamin bentuk distribusi normal.

## BAB IV — Pemodelan

### IV.1 — K-Means K=2

Input hanya dua kolom Z-score: X=minat kejuruan dan Y=rerata empat mapel. Fit K-Means pada 141 data latih, n_init=10, random_state=42. Tetapkan nomor semantik berdasarkan jumlah koordinat centroid Z-score yang lebih tinggi sebagai kode 1 (Sesuai), lainnya kode 0 (Tidak Sesuai). Ini adalah konvensi operasional, bukan penilaian terverifikasi atas salah/benar jurusan.

K=2 dipilih karena tujuan penelitian mengelompokkan dua profil. Tidak diklaim sebagai puncak silhouette seluruh kemungkinan K. Siswa ditempatkan berdasarkan centroid terdekat dalam ruang terstandarisasi; grafik menampilkan satuan asli X (7–35) dan Y (0–100).

{table(['Kode / label','Assignment seluruh 177','Centroid X dari data latih','Centroid Y dari data latih'],[
['0 / Tidak Sesuai',km['cluster_counts']['0'],f"{c0['x']:.4f}",f"{c0['y']:.4f}"],
['1 / Sesuai',km['cluster_counts']['1'],f"{c1['x']:.4f}",f"{c1['y']:.4f}"]])}

Kedua kelompok berbeda terutama pada minat; centroid Sesuai tidak memiliki rapor lebih tinggi daripada kelompok lain. Seorang anggota Sesuai dapat memiliki rapor rendah. Jika dibutuhkan syarat mutlak X dan Y melewati ambang tertentu, itu memerlukan aturan domain tambahan yang belum diterapkan di sini.

### IV.2 — Random Forest

RF memakai 10 fitur Z-score (empat rapor dan enam RIASEC) ditambah satu kode klaster biner. Parameter tetap 100 pohon, Gini, bootstrap, random_state=42. Klaster merupakan fitur tambahan, bukan saklar yang memaksa pilihan program studi. Target adalah lima rumpun proxy proyek; kesesuaiannya dengan kategori resmi seleksi perguruan tinggi tidak diasumsikan.

RF scikit-learn mengagregasi probabilitas daun seluruh pohon. Voting keras ditampilkan sebagai informasi tambahan dan tidak didefinisikan sebagai aturan umum predict_proba. Confidence bukan persentase keberhasilan kuliah dan belum dikalibrasi.

## BAB V — Evaluasi dan batas hasil

{table(['Metrik','Nilai','Cakupan'],[
['Inertia K-Means',f"{km['inertia_train']:.6f}",'Hanya 141 data yang dipakai fit'],
['Silhouette latih',f"{km['silhouette_train']:.6f}",'141 data latih'],
['Silhouette seluruh assignment',f"{km['silhouette_all_assigned']:.6f}",'177 siswa, centroid dipelajari dari latih'],
['Davies–Bouldin seluruh assignment',f"{km['davies_bouldin_all_assigned']:.6f}",'177 assignment'],
['Akurasi latih RF',f"{m['train_accuracy']:.2%}",'141 siswa'],
['Akurasi holdout RF',f"{m['test_accuracy']:.2%}",f"{m['test_correct']}/{m['test_rows']} terhadap target proxy"],
['Macro F1 kelas dengan support uji',f"{m['macro_evaluated_classes']['f1-score']:.2%}",'Empat kelas terwakili di uji'],
['RF tanpa fitur klaster',f"{m['rf_without_cluster_accuracy']:.2%}",'Split dan preprocessing latih yang sama'],
['Baseline kelas mayoritas',f"{m['baseline_majority_accuracy']:.2%}",'Prediksi kelas terbanyak latih'],
['Repeated stratified 2-fold, 5 ulang',f"{m['cv_accuracy_mean']:.2%} ± {m['cv_accuracy_std_descriptive']*100:.2f} poin",'88/89 data latih per fold; std deskriptif, bukan CI']])}

Per kelas:
{table(['Rumpun','Support uji','Precision','Recall','F1'],per_class)}

Sosial hanya memiliki dua anggota di seluruh data dan nol contoh pada holdout ini. Kemampuan generalisasi kelas tersebut belum terukur. Dua fold CV dipakai karena kelas terkecil hanya dua anggota; jumlahnya tetap belum memadai untuk klaim kuat. Selisih hybrid dan RF tanpa cluster belum membuktikan peningkatan oleh K-Means.

Importance fitur klaster adalah {m['feature_importance']['Klaster_KMeans']:.2%}, bukan 10,1%. Importance adalah ukuran prediktif relatif, bukan pengaruh kausal. Seluruh angka importance tersedia dalam feature_importance_177.csv.

Histogram mencakup seluruh {m['test_rows']} prediksi, termasuk {m['confidence_below_50']} prediksi confidence <50%. Grafik pohon dibaca langsung dari estimator, dengan samples unik dibedakan dari bobot bootstrap. Gambar kedalaman terbatas diberi label cuplikan.

## BAB VI — Implementasi aplikasi dan kesimpulan

Python API, demo CLI, Streamlit, dan Flutter memakai bundle/kontrak versi yang sama. API mengirim skor X–Y siswa, dua centroid, kedua jarak Z-score, kode klaster, probabilitas, voting, dan hubungan rumpun–jurusan. Hubungan Linier/Lintas bidang dihitung dari rumpun hasil dibandingkan jurusan asal, bukan otomatis dari kode klaster.

Tampilan yang diacu: profil enam skor RIASEC → Tahap 1 K-Means (X,Y,dua centroid,status) → Tahap 2 RF (rumpun,confidence,probabilitas,voting) → contoh program studi referensi dan diskusi BK. Tampilan hasil lama tetap menggunakan metadata versi yang tersimpan; jangan mencampur centroid versi baru dengan hasil lama.

Kesimpulan yang didukung: pipeline dapat mengintegrasikan 177 data, membentuk dua kelompok menurut rancangan, dan mereproduksi target proxy dengan akurasi holdout {m['test_accuracy']:.2%}. Klaim keberhasilan konseling, salah jurusan, validitas keseluruhan instrumen, dan peningkatan oleh hybrid membutuhkan bukti tambahan. Hasil ini merupakan prototipe pendukung eksplorasi studi lanjut.

## Pemetaan kode dan reproduksi

{source_map}

Skrip per bab merupakan entry point; rumus dan implementasi bersama berada dalam research_pipeline.py agar ekspor model, evaluasi, dan grafik tidak menggunakan parameter berbeda. Jalankan dari root repo: python tools/run_research_revision.py. Pelatihan ulang memerlukan dua sumber data mentah privat; demo/API dapat memakai bundle yang sudah tersedia. Untuk tes: python -m unittest discover -s pipeline_177_data_mining -p test_research_pipeline.py; python pipeline_177_data_mining/test_inference_engine.py.

## Lampiran yang digunakan

Gunakan delapan grafik baru di 1_Diagram_dan_Gambar_Laporan, tabel anonim di 2_Tabel_Pengujian_dan_Hasil_Metrik, dan workbook diagnostik item-rest. Versi draf dengan angka 117/60, 91/86, centroid lama, klaim K=2 paling tinggi, atau klaim kontribusi klaster 10,1% tidak menjadi acuan model 1.1.0.

Referensi metode implementasi: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage ; https://scikit-learn.org/stable/modules/clustering.html#k-means ; https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html .
'''
    folders=['0_Source_Code_Per_Bab_Laporan','1_Diagram_dan_Gambar_Laporan','2_Tabel_Pengujian_dan_Hasil_Metrik','3_Panduan_Narasi_dan_Angka_Resmi_Laporan','5_Petunjuk_Tautan_dan_Demo_Aplikasi']
    for folder in folders:(PACKAGE/folder).mkdir(parents=True,exist_ok=True)
    for p in BASE.glob('*.py'):shutil.copy2(p,PACKAGE/folders[0]/p.name)
    for name in ['app.py','demo_aplikasi_nalara.py','uji_pembuktian_akademik.py']:
        shutil.copy2(ROOT/name,PACKAGE/folders[0]/name)
    images=['scatter_kmeans_2d_177.png','evaluasi_silhouette_kmeans_177.png','confusion_matrix_rf_177.png',
            'feature_importance_rf_177.png','hasil_voting_nyata_100_pohon_177.png',
            'struktur_pohon_keputusan_asli_rf_177.png','visualisasi_pohon_keputusan_rf_177.png','diagram_arsitektur_random_forest.png']
    for name in images:shutil.copy2(OUT/name,PACKAGE/folders[1]/name)
    for p in (BASE/'output_eda').glob('eda_*.png'):shutil.copy2(p,PACKAGE/folders[1]/p.name)
    for name in ['dataset_anonim_177.csv','dataset_engineered_177.csv','dataset_siap_model_177.csv','dataset_terklaster_177.csv','hasil_prediksi_rf_177.csv']:
        shutil.copy2(BASE/name,PACKAGE/folders[2]/name)
    for name in ['metrics_177.json','feature_importance_177.csv','rekapitulasi_voting_100_pohon_177.csv','split_177.json']:
        shutil.copy2(OUT/name,PACKAGE/folders[2]/name)
    for name in ['diagnostik_item_177.csv','reliabilitas_177.csv','tabel_korelasi_rapor_riasec.csv','Hasil_Uji_Validitas_dan_Reliabilitas_Instrumen_RIASEC.xlsx']:
        shutil.copy2(BASE/'output_eda'/name,PACKAGE/folders[2]/name)
    guide=PACKAGE/folders[3]
    (guide/'RINGKASAN_LAPORAN_DAN_KONSULTASI_DOSEN.md').write_text(text,encoding='utf-8')
    (ROOT/'RINGKASAN_LAPORAN_DAN_KONSULTASI_DOSEN.md').write_text(text,encoding='utf-8')
    write_docx(guide/'ringkasan project datmin_revisi_v2.docx',text)
    notes=f'''# Penggantian bahan tim — model 1.1.0

Gunakan paket v2 ini untuk naskah akhir. Dokumen Word asli tidak ditimpa.

- Pertahankan: 177 siswa, 42 butir, X minat kejuruan, Y rerata empat mapel, K=2, RF100.
- Ganti Bab III: split sebelum fit scaler/K-Means. Sumber fitur/target ditulis sebagai target proxy aturan.
- Ganti Bab IV: Sesuai {km['cluster_counts']['1']}, Tidak Sesuai {km['cluster_counts']['0']}; centroid latih X/Y C0=({c0['x']:.4f},{c0['y']:.4f}), C1=({c1['x']:.4f},{c1['y']:.4f}).
- Ganti Bab V: inertia latih {km['inertia_train']:.6f}; silhouette latih {km['silhouette_train']:.6f}; silhouette 177 assignment {km['silhouette_all_assigned']:.6f}; akurasi tetap {m['test_accuracy']:.2%}, support Sosial=0.
- Ganti seluruh grafik pohon, voting, scatter, silhouette, confusion matrix, importance, dan arsitektur dengan berkas v2.
- Perbarui batas instrumen: alpha Artistic rendah, {instrument['item_rest_flagged']} butir perlu telaah item-rest; hapus klaim otomatis 100% valid.
- Selaraskan kesimpulan: prototipe pendukung konseling; manfaat nyata dan kecocokan jurusan belum divalidasi independen.
- Perbarui screenshot aplikasi setelah bundle 1.1.0 dan UI hasil dua tahap digunakan.
- Naskah yang sudah disalin tim perlu diperbarui manual sesuai daftar ini. Paket tidak mengubah salinan milik teman.

File identitas NIS/Nama tidak disertakan. Semua berkas kode di folder 0 adalah salinan untuk kutipan; jalankan program dari struktur repo utuh.
'''
    (PACKAGE/'MULAI_DI_SINI.md').write_text(notes,encoding='utf-8')
    (guide/'DAFTAR_PERUBAHAN_UNTUK_TIM.md').write_text(notes,encoding='utf-8')
    (PACKAGE/folders[0]/'PANDUAN_LETAK_KODE_PER_SUB_BAB.txt').write_text(source_map,encoding='utf-8')
    (PACKAGE/folders[4]/'PETUNJUK_GITHUB_DAN_DEMO_STREAMLIT.txt').write_text(
        'Repo: https://github.com/auriellifta/Project-Data-Mining\nVersi lokal revisi: 1.1.0; status publikasi mengikuti commit aktual.\nDemo: streamlit run app.py\nAPI: python pipeline_177_data_mining/api_server.py\nReproduksi: python tools/run_research_revision.py\n',encoding='utf-8')
    readme=f'''# NALARA — Penelitian dan integrasi SMK

Model **{m['model_version']}**: 177 siswa; 42 butir; K-Means 2D **K=2** (X=minat kejuruan, Y=rerata empat mapel); RF100.

Scaler, K-Means, dan RF fit hanya pada **141 data latih**. Bagian uji 36 siswa ditransformasi dengan parameter latih.
Assignment seluruh data: Sesuai **{km['cluster_counts']['1']}**, Tidak Sesuai **{km['cluster_counts']['0']}**. Label adalah kategori operasional kelompok, bukan diagnosis salah jurusan atau aturan dua ambang.

Akurasi terhadap target proxy aturan **{m['test_accuracy']:.2%} ({m['test_correct']}/36)**. Sosial tidak memiliki contoh pada holdout, sehingga kinerjanya belum terukur. RF tanpa cluster pada split sama **{m['rf_without_cluster_accuracy']:.2%}**; peningkatan oleh hybrid belum terbukti.

## Jalankan

```powershell
python tools/run_research_revision.py
python -m unittest discover -s pipeline_177_data_mining -p test_research_pipeline.py
python pipeline_177_data_mining/test_inference_engine.py
streamlit run app.py
```

Pelatihan ulang memerlukan sumber data mentah privat. API menggunakan NALARA_API_KEY; lihat docs/PYTHON_API_DOCKER_RUNBOOK.md. Semua inferensi menggunakan pipeline_bundle_177.joblib, tanpa menghitung ulang mean/std atau memakai centroid konstan.

## Bahan laporan

[Ringkasan revisi](RINGKASAN_LAPORAN_DAN_KONSULTASI_DOSEN.md) · Paket lokal: Bahan_Laporan_Akhir_Untuk_Tim_v2/MULAI_DI_SINI.md. Paket lama adalah arsip. Salinan kode per bab memakai research_pipeline.py sebagai sumber rumus bersama.

Diagnostik instrumen menyediakan item-rest dan alpha; {instrument['item_rest_flagged']} butir memerlukan telaah. Korelasi tidak memvalidasi otomatis seluruh instrumen maupun target RF.

## Sumber resmi metode

[Pencegahan data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means), [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    old=ROOT/'Bahan_Laporan_Akhir_Untuk_Tim'
    if old.exists():(old/'PAKET_LAMA_GUNAKAN_REVISI_V2.txt').write_text('Paket ini merupakan arsip sebelum koreksi metodologi. Gunakan ../Bahan_Laporan_Akhir_Untuk_Tim_v2/MULAI_DI_SINI.md dan naskah revisi v2.\n',encoding='utf-8')
    hashes={p.relative_to(PACKAGE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in PACKAGE.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    (PACKAGE/'MANIFEST.json').write_text(json.dumps({'model_version':m['model_version'],'sha256':hashes},indent=2),encoding='utf-8')
    print('Paket tim diperbarui:',PACKAGE)


if __name__=='__main__':export()
