# NALARA — Nalar & Arah Karier

> **Decision Support System (DSS) Rekomendasi Program Studi Perguruan Tinggi bagi Siswa SMK Menggunakan Pendekatan Two-Stage Hybrid Machine Learning (K-Means Clustering 2D & Random Forest Classifier)**

---

## 📌 Ringkasan Proyek

Proyek ini dikembangkan untuk memenuhi Tugas Akhir Mata Kuliah **Penambangan Data (INF11121)**, Program Studi Teknik Informatika, Fakultas Teknik dan Teknologi Kemaritiman, **Universitas Maritim Raja Ali Haji (UMRAH)**.

NALARA dirancang untuk menjawab permasalahan *major mismatch* (kesenjangan minat dan studi lanjut) pada siswa SMK dengan mengintegrasikan data rekam kognitif akademik (**nilai rapor semester**) dan asesmen minat psikometri terstandarisasi (**Holland RIASEC**).

* **Studi Kasus:** SMK Negeri 1 Bintan Timur, Kepulauan Riau
* **Ukuran Sampel:** 177 Responden Aktif (Kelas X, XI, dan XII)
* **Konsentrasi Keahlian:** RPL, AKL, PH, DKV, dan Tata Busana

---

## 🏗️ Metodologi & Arsitektur Model

Sistem menerapkan arsitektur dua tahap terpadu (*Two-Stage Hybrid Architecture*):

```
[Input Data Siswa]
  ├── Nilai Rapor (Matematika, B. Indonesia, B. Inggris, Produktif)
  └── Asesmen Minat Holland RIASEC (R, I, A, S, E, C)
          │
          ▼
[TAHAP 1: K-Means Clustering 2D (K=2)]
  ├── Sumbu X: Skor Minat Kejuruan Spesifik
  └── Sumbu Y: Rata-Rata 4 Mapel Kunci Rapor
          │
          ├──> Centroid 1 (Kongruen / Sesuai Jurusan)
          └──> Centroid 0 (Diskongruen / Mismatch)
          │
          ▼ [Status Kesesuaian K-Means (0/1)]
          │
[TAHAP 2: Random Forest Classifier (100 Trees)]
  ├── 11 Fitur Masukan Terstandarisasi Z-Score
  └── Mekanisme Majority Voting (100 Estimators)
          │
          ▼
[Output Keputusan Sistem]
  ├── Rekomendasi Linier Sejalur SMK (Bagi Siswa Kongruen)
  └── Rekomendasi Penyelamatan Lintas Minat (Bagi Siswa Diskongruen)
```

---

## 📊 Hasil Evaluasi Kinerja Model

| Metrik Evaluasi | Nilai Kinerja | Interpretasi Akademis |
| :--- | :---: | :--- |
| **Silhouette Score K-Means** | **0.3213** | Struktur klaster terpisah dengan baik (*K=2*) |
| **Inersia K-Means (SSE)** | **236.46** | Konvergensi klaster optimal |
| **Akurasi Pelatihan RF** | **100.00%** | Kemampuan *fitting* sempurna pada data latih |
| **Akurasi Pengujian RF** | **86.11%** | 31 dari 36 data uji independen diprediksi tepat |
| **Rata-Rata Precision (Macro)** | **87.09%** | Ketepatan tinggi per rumpun prodi |
| **Rata-Rata Recall (Macro)** | **85.73%** | Sensitivitas tinggi terhadap kelas minoritas |
| **Rata-Rata F1-Score (Macro)** | **85.34%** | Keseimbangan presisi dan recall |
| **Rata-Rata Konsensus Voting** | **71.86%** | Rata-rata 71 dari 100 pohon sepakat satu suara |

---

## 📂 Struktur Berkas Repositori

```
├── app.py                                   # Aplikasi Web Dashboard DSS (Streamlit)
├── demo_aplikasi_nalara.py                  # Skrip Inferensi Cepat CLI
├── uji_pembuktian_akademik.py               # Uji Validitas, Reliabilitas Cronbach, & Korelasi
│
├── pipeline_177_data_mining/                # Pipeline Analisis Saintifik Per Bab
│   ├── step1_data_understanding_177.py      # Bab II: Eksplorasi Data & 5 Grafik EDA
│   ├── step2_preprocessing_177.py           # Bab III: Data Preprocessing Terpadu
│   ├── bab3_sub1_data_cleaning_177.py       # Bab III.1: Data Cleaning & Skala Likert
│   ├── bab3_sub2_data_selection_177.py      # Bab III.2: Seleksi 4 Mapel Kunci
│   ├── bab3_sub3_data_integration_177.py    # Bab III.3: Integrasi Data via NIS
│   ├── bab3_sub4_data_anonymization_177.py  # Bab III.4: Anonimisasi & Proteksi PII
│   ├── bab3_sub5_feature_engineering_177.py # Bab III.5: Feature & Target Engineering
│   ├── bab3_sub6_data_transformation_177.py # Bab III.6: Z-Score StandardScaler
│   ├── bab4_sub1_kmeans_177.py              # Bab IV.1: Pemodelan K-Means 2D
│   ├── bab4_sub2_random_forest_177.py       # Bab IV.2: Pemodelan Random Forest 100 Pohon
│   ├── buat_diagram_arsitektur_random_forest.py # Pembuat Diagram Arsitektur RF
│   ├── bab5_sub1_evaluasi_kmeans_177.py     # Bab V.1: Evaluasi Silhouette K-Means
│   ├── bab5_sub2_evaluasi_random_forest_177.py # Bab V.2: Evaluasi Confusion Matrix & Feature Importance
│   ├── visualisasi_voting_nyata_100_pohon.py   # Bab V.3: Sebaran Voting 100 Pohon Data Uji
│   └── gambar_pohon_keputusan_asli.py       # Bab V.3: Diagram Struktur Percabangan Tree #1
│
├── output_eda/                              # 5 Grafik Hasil Visualisasi Bab II
└── output_model/                            # Model Terlatih (.joblib) & Visualisasi Bab IV - V
```

---

## 🚀 Panduan Menjalankan Aplikasi

### 1. Prasyarat Lingkungan
Pastikan Python 3.10 atau versi lebih baru telah terpasang. Pasang dependensi yang diperlukan:
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib openpyxl scipy
```

### 2. Menjalankan Dashboard Web Interaktif (Streamlit)
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

### 3. Menjalankan Demonstrasi Inferensi Cepat (Terminal CLI)
```bash
python demo_aplikasi_nalara.py
```

### 4. Menjalankan Uji Pembuktian Psikometri & Akademik
```bash
python uji_pembuktian_akademik.py
```

---

## 👥 Tim Pengembang

* **Mata Kuliah:** Penambangan Data (INF11121)
* **Semester / Tahun Akademik:** Ganjil 2026/2027
* **Program Studi:** S1 Teknik Informatika
* **Institusi:** Fakultas Teknik dan Teknologi Kemaritiman, Universitas Maritim Raja Ali Haji (UMRAH)
