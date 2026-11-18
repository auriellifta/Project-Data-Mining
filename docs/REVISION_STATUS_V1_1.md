# Status revisi NALARA / Data Mining 1.1.0

19 November 2026. Perubahan lokal pada kedua checkout; belum ada commit, push, atau deployment remote.

## Rancangan dan hasil

- Tetap 177 siswa, 42 butir, X skor dimensi RIASEC terkait jurusan, Y rerata empat mapel, K=2, RF100.
- Scaler, K-Means, dan RF dipelajari hanya dari 141 bagian latih; 36 bagian uji ditransformasi dengan parameter latih.
- Assignment 177 siswa pada model revisi: **Sesuai 87**, **Tidak Sesuai 90**. Perubahan dari 91/86 berasal dari estimator yang kini hanya melihat data latih. Nilai mentah dan target proxy tidak diubah.
- Akurasi holdout **31/36 =86,11%** terhadap target proxy aturan. Sosial memiliki support uji nol. Keunggulan hybrid dan kecocokan karier nyata belum dibuktikan oleh angka ini.
- Silhouette latih **0,340531**; silhouette seluruh assignment **0,330059**; inertia latih **181,856130**. Cakupan setiap metrik dinyatakan dalam naskah.
- Sesuai/Tidak Sesuai adalah label operasional dua kelompok, bukan syarat mutlak dua ambang minimum. Instrumen tetap 42 butir; diagnostik item-rest menandai lima butir untuk ditelaah dan alpha Artistic rendah diungkapkan.

## Perubahan aplikasi

- Flutter SMK menampilkan Tahap 1 K-Means dan Tahap 2 RF secara terpisah.
- Panel scatter menampilkan titik siswa, centroid C0/C1, X mentah 7–35, Y mentah 0–100, dan dua jarak dalam ruang Z-score.
- API dan snapshot database membawa geometri dan hubungan Linier/Lintas bidang. Hubungan dihitung dari jurusan asal dan rumpun hasil, bukan otomatis dari kode klaster.
- Hasil lama tanpa geometri tidak digambar menggunakan centroid baru. Label historis dan alur Mahasiswa tetap ditangani melalui versi lama.
- Pilihan Likert pada instrumen SMK memakai lima teks resmi penelitian.
- Streamlit dan CLI memakai bundle yang sama dengan API, tanpa mean/std atau centroid manual.

## Bukti verifikasi

| Pemeriksaan | Hasil |
| --- | --- |
| Pipeline dari sumber privat | Berhasil; validasi silang nama/NIS 177/177; nilai dan target tidak berubah |
| Regresi metodologi Python | 3 tes lulus, termasuk perubahan ekstrem data uji tidak memengaruhi scaler/K-Means/RF yang dilatih |
| Parity engine | 36/36 sama dengan prediksi tersimpan; ini parity, bukan akurasi 100% |
| Validasi engine dan Flask | 16 kasus batas dan 7 skenario HTTP lulus; termasuk geometri API model 1.1.0 |
| Docker API 1.1.0 | Build, health, 401, 422, 200, X/Y, dan voting lulus; container dibersihkan |
| SQL PostgreSQL 16.16 | Schema dua kali, 25 migrasi masing-masing dua kali, regresi promotions/career lulus |
| SQL revisi | Respons beda versi ditolak; geometri dan hubungan bertahan pada snapshot reload |
| Dua koneksi nyata | Lock teramati dengan blocking PID; 409, takeover, token lama ditolak, idempoten, nol sisa fixture |
| Flutter analyze | No issues found pada tujuh berkas terkait |
| Flutter test | 31/31 lulus, termasuk widget lebar ponsel dan hasil historis tanpa geometri |
| Flutter Web | Build berhasil, Wasm dry run berhasil; tanpa APK; 353,7 detik |
| Paket tim | 56 hash cocok; CSV tanpa kolom identitas langsung; DOCX dapat diparse dan memuat tabel |

Uji browser penuh terhadap stack Supabase + Edge + Python versi revisi belum diulang. Build dan tes widget tidak diklaim sebagai bukti E2E browser. Peringatan scikit-learn tentang nama fitur array masih dapat muncul pada pengujian Python; urutan fitur dan hasil prediksi diverifikasi.

## Berkas untuk tim

- `Bahan_Laporan_Akhir_Untuk_Tim_v2.zip`
- `Bahan_Laporan_Akhir_Untuk_Tim_v2/MULAI_DI_SINI.md`
- `Bahan_Laporan_Akhir_Untuk_Tim_v2/3_Panduan_Narasi_dan_Angka_Resmi_Laporan/ringkasan project datmin_revisi_v2.docx`
- `RINGKASAN_LAPORAN_DAN_KONSULTASI_DOSEN.md`

Dokumen Word asli dan paket lama dipertahankan. Paket lama diberi penanda pengganti v2. Tim perlu mengganti salinan naskah/gambar yang sudah mereka gunakan sesuai daftar perubahan.

CSV terintegrasi yang memuat Nama/NIS sudah dikeluarkan dari index Git, diabaikan untuk perubahan berikutnya, dan tetap disimpan lokal. **Langkah ini tidak menghapus data identitas dari riwayat commit lama maupun remote yang belum diperbarui.** Penanganan riwayat membutuhkan koordinasi terpisah; tidak ada rewrite/force push dari revisi ini.

Untuk menyalakan backend revisi, gunakan bundle/API 1.1.0, Edge Function revisi, dan `career_prediction_hybrid.sql` diikuti `career_prediction_model_v1_1.sql`. Penerapan remote masih ditunda sesuai arahan pengguna.
