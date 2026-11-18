# NALARA — Penelitian dan integrasi SMK

Model **1.1.0**: 177 siswa; 42 butir; K-Means 2D **K=2** (X=minat kejuruan, Y=rerata empat mapel); RF100.

Scaler, K-Means, dan RF fit hanya pada **141 data latih**. Bagian uji 36 siswa ditransformasi dengan parameter latih.
Assignment seluruh data: Sesuai **87**, Tidak Sesuai **90**. Label adalah kategori operasional kelompok, bukan diagnosis salah jurusan atau aturan dua ambang.

Akurasi terhadap target proxy aturan **86.11% (31/36)**. Sosial tidak memiliki contoh pada holdout, sehingga kinerjanya belum terukur. RF tanpa cluster pada split sama **88.89%**; peningkatan oleh hybrid belum terbukti.

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

Diagnostik instrumen menyediakan item-rest dan alpha; 5 butir memerlukan telaah. Korelasi tidak memvalidasi otomatis seluruh instrumen maupun target RF.

## Sumber resmi metode

[Pencegahan data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means), [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
