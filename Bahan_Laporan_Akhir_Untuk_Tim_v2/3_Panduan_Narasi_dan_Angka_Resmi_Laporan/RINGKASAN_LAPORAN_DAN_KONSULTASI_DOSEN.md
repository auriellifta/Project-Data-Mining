# RINGKASAN PROYEK DATA MINING NALARA — REVISI V2

Versi model: 1.1.0 · Pipeline: pipeline_177_smk_train_only_v1_1 · 19 November 2026.
Sumber angka: output_model/metrics_177.json. Naskah ini menggantikan angka dan klaim draf sebelumnya.

## BAB I — Tujuan dan batas penelitian

NALARA mengeksplorasi profil minat kejuruan dan kemampuan akademik 177 siswa aktif SMKN 1 Bintan Timur, lalu memprediksi lima rumpun studi lanjut. Hasil digunakan sebagai bahan diskusi konseling. Data berasal dari siswa kelas X, XI, XII, bukan rekam longitudinal 177 siswa selama tiga tahun.

Rancangan dipertahankan: 42 butir, skor komposit enam dimensi 7–35, K-Means dua dimensi dengan K=2, dan Random Forest 100 pohon. Label Sesuai/Tidak Sesuai adalah definisi operasional kelompok relatif terhadap dua fitur; keanggotaan tidak mensyaratkan dua batas minimum dan tidak membuktikan seseorang salah jurusan.

## BAB II — Data understanding dan diagnostik instrumen

Sumber terdiri dari 29 leger semester genap 2026 dan satu kuesioner 177 responden. Audit integritas menemukan seluruh 177 NIS dan nama setelah normalisasi cocok; tidak ada jawaban 42 butir yang kosong/tidak dikenal. Ekstraksi mapel produktif mengikuti legenda mapel tiap leger; bobot setara mapel produktif perlu dikonfirmasi sebagai keputusan domain.

Komposisi jurusan: RPL 41, AKT/AKL 39, PH 38, DKV 37, TB 22. NIS/Nama hanya dipakai dalam proses integrasi privat; tabel model dan paket tim memakai ID pseudonim.

Alpha per dimensi:
| Dimensi | Cronbach alpha |
| --- | --- |
| Realistic | 0.6291 |
| Investigative | 0.7332 |
| Artistic | 0.3210 |
| Social | 0.5976 |
| Enterprising | 0.5703 |
| Conventional | 0.6329 |

Alpha Artistic yang rendah diungkapkan sebagai keterbatasan. Standardisasi tidak memperbaiki reliabilitas. Korelasi item-total memasukkan butirnya sendiri, sehingga disertakan item-rest yang mengeluarkan butir tersebut. Sebanyak 5 butir perlu ditelaah berdasar acuan korelasi item-rest eksploratif; 42 butir tetap digunakan pada instrumen versi ini demi konsistensi prototipe, dengan keterbatasan tersebut dinyatakan terbuka. Ini bukan kesimpulan bahwa seluruh instrumen valid secara isi/konstruk. Validasi pakar dan kajian instrumen lanjutan belum dibuktikan oleh perhitungan korelasi saja.

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

| Kode / label | Assignment seluruh 177 | Centroid X dari data latih | Centroid Y dari data latih |
| --- | --- | --- | --- |
| 0 / Tidak Sesuai | 90 | 24.4638 | 84.6250 |
| 1 / Sesuai | 87 | 28.6250 | 82.9688 |

Kedua kelompok berbeda terutama pada minat; centroid Sesuai tidak memiliki rapor lebih tinggi daripada kelompok lain. Seorang anggota Sesuai dapat memiliki rapor rendah. Jika dibutuhkan syarat mutlak X dan Y melewati ambang tertentu, itu memerlukan aturan domain tambahan yang belum diterapkan di sini.

### IV.2 — Random Forest

RF memakai 10 fitur Z-score (empat rapor dan enam RIASEC) ditambah satu kode klaster biner. Parameter tetap 100 pohon, Gini, bootstrap, random_state=42. Klaster merupakan fitur tambahan, bukan saklar yang memaksa pilihan program studi. Target adalah lima rumpun proxy proyek; kesesuaiannya dengan kategori resmi seleksi perguruan tinggi tidak diasumsikan.

RF scikit-learn mengagregasi probabilitas daun seluruh pohon. Voting keras ditampilkan sebagai informasi tambahan dan tidak didefinisikan sebagai aturan umum predict_proba. Confidence bukan persentase keberhasilan kuliah dan belum dikalibrasi.

## BAB V — Evaluasi dan batas hasil

| Metrik | Nilai | Cakupan |
| --- | --- | --- |
| Inertia K-Means | 181.856130 | Hanya 141 data yang dipakai fit |
| Silhouette latih | 0.340531 | 141 data latih |
| Silhouette seluruh assignment | 0.330059 | 177 siswa, centroid dipelajari dari latih |
| Davies–Bouldin seluruh assignment | 1.165417 | 177 assignment |
| Akurasi latih RF | 100.00% | 141 siswa |
| Akurasi holdout RF | 86.11% | 31/36 terhadap target proxy |
| Macro F1 kelas dengan support uji | 85.34% | Empat kelas terwakili di uji |
| RF tanpa fitur klaster | 88.89% | Split dan preprocessing latih yang sama |
| Baseline kelas mayoritas | 30.56% | Prediksi kelas terbanyak latih |
| Repeated stratified 2-fold, 5 ulang | 79.99% ± 4.71 poin | 88/89 data latih per fold; std deskriptif, bukan CI |

Per kelas:
| Rumpun | Support uji | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Ekonomi, Bisnis & Akuntansi | 11 | 100.00% | 81.82% | 90.00% |
| Pariwisata & Hospitality | 6 | 71.43% | 83.33% | 76.92% |
| Sains & Teknologi Informasi | 9 | 100.00% | 77.78% | 87.50% |
| Seni, Desain & Industri Kreatif | 10 | 76.92% | 100.00% | 86.96% |
| Sosial, Humaniora & Pendidikan | 0 | Belum terukur | Belum terukur | Belum terukur |

Sosial hanya memiliki dua anggota di seluruh data dan nol contoh pada holdout ini. Kemampuan generalisasi kelas tersebut belum terukur. Dua fold CV dipakai karena kelas terkecil hanya dua anggota; jumlahnya tetap belum memadai untuk klaim kuat. Selisih hybrid dan RF tanpa cluster belum membuktikan peningkatan oleh K-Means.

Importance fitur klaster adalah 0.67%, bukan 10,1%. Importance adalah ukuran prediktif relatif, bukan pengaruh kausal. Seluruh angka importance tersedia dalam feature_importance_177.csv.

Histogram mencakup seluruh 36 prediksi, termasuk 8 prediksi confidence <50%. Grafik pohon dibaca langsung dari estimator, dengan samples unik dibedakan dari bobot bootstrap. Gambar kedalaman terbatas diberi label cuplikan.

## BAB VI — Implementasi aplikasi dan kesimpulan

Python API, demo CLI, Streamlit, dan Flutter memakai bundle/kontrak versi yang sama. API mengirim skor X–Y siswa, dua centroid, kedua jarak Z-score, kode klaster, probabilitas, voting, dan hubungan rumpun–jurusan. Hubungan Linier/Lintas bidang dihitung dari rumpun hasil dibandingkan jurusan asal, bukan otomatis dari kode klaster.

Tampilan yang diacu: profil enam skor RIASEC → Tahap 1 K-Means (X,Y,dua centroid,status) → Tahap 2 RF (rumpun,confidence,probabilitas,voting) → contoh program studi referensi dan diskusi BK. Tampilan hasil lama tetap menggunakan metadata versi yang tersimpan; jangan mencampur centroid versi baru dengan hasil lama.

Kesimpulan yang didukung: pipeline dapat mengintegrasikan 177 data, membentuk dua kelompok menurut rancangan, dan mereproduksi target proxy dengan akurasi holdout 86.11%. Klaim keberhasilan konseling, salah jurusan, validitas keseluruhan instrumen, dan peningkatan oleh hybrid membutuhkan bukti tambahan. Hasil ini merupakan prototipe pendukung eksplorasi studi lanjut.

## Pemetaan kode dan reproduksi

| Subbab | Sumber utama |
| --- | --- |
| II | step1_data_understanding_177.py; instrument_validation.py |
| III.1–III.5 | bab3_sub1 ... bab3_sub5 (pembersihan, integrasi, pseudonim, fitur, target proxy) |
| III.6 | bab3_sub6_data_transformation_177.py → research_pipeline.transform_dataset |
| IV.1 | bab4_sub1_kmeans_177.py → research_pipeline.fit_kmeans |
| IV.2 | bab4_sub2_random_forest_177.py → research_pipeline.fit_forest |
| V | bab5_sub1/bab5_sub2; research_pipeline.evaluate dan fungsi plot |
| VI | inference_engine.py, api_server.py, app.py, demo_aplikasi_nalara.py |

Skrip per bab merupakan entry point; rumus dan implementasi bersama berada dalam research_pipeline.py agar ekspor model, evaluasi, dan grafik tidak menggunakan parameter berbeda. Jalankan dari root repo: python tools/run_research_revision.py. Pelatihan ulang memerlukan dua sumber data mentah privat; demo/API dapat memakai bundle yang sudah tersedia. Untuk tes: python -m unittest discover -s pipeline_177_data_mining -p test_research_pipeline.py; python pipeline_177_data_mining/test_inference_engine.py.

## Lampiran yang digunakan

Gunakan delapan grafik baru di 1_Diagram_dan_Gambar_Laporan, tabel anonim di 2_Tabel_Pengujian_dan_Hasil_Metrik, dan workbook diagnostik item-rest. Versi draf dengan angka 117/60, 91/86, centroid lama, klaim K=2 paling tinggi, atau klaim kontribusi klaster 10,1% tidak menjadi acuan model 1.1.0.

Referensi metode implementasi: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage ; https://scikit-learn.org/stable/modules/clustering.html#k-means ; https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html .
