# Panduan diagram laporan NALARA — model 1.1.0

Pembaruan ini memperbaiki bentuk dan keterbacaan gambar. Data, pembagian 141 latih / 36 uji, scaler, K-Means K=2, RF100, dan metrik tidak dilatih atau dihitung ulang dengan rancangan berbeda.

Judul utama diagram ditempatkan di atas dan rata tengah. Catatan teknis tentang agregasi probabilitas scikit-learn dan kalibrasi confidence disampaikan dalam dokumen pendamping, tidak sebagai catatan kaki pada grafik voting.

## Pilihan distribusi untuk tim

ZIP disaring melalui `tools/report_image_selection.json`. Pilihan yang dikonfirmasi untuk lampiran tim adalah diagram ensemble bernomor `diagram_pohon_lingkaran_rf.png` dan cuplikan pohon `struktur_pohon_keputusan_asli_rf_177.png`. Grafik EDA, scatter, silhouette, confusion matrix, feature importance, dan voting tetap disertakan sebagai bahan analisis. Tidak ada gambar cadangan yang dihapus dari folder lokal; daftar berkas yang benar-benar disertakan dalam ZIP tercatat di `MANIFEST.json` dan `MULAI_DI_SINI.md`.

Diagram RF lain, gambar pohon penuh, SVG, dan salinan JPG tersedia lokal. Paket tim dan variasi lokal dikecualikan dari pelacakan Git; repositori menyimpan kode reproduksi dan gambar penelitian yang diperlukan. Melepas pelacakan tidak menghapus commit lama.

## Pilih gambar sesuai tujuan

| Berkas | Tujuan dan cara membaca |
| --- | --- |
| `scatter_kmeans_2d_177.png` | Posisi seluruh 177 siswa dan dua centroid. Rentang tampilan mengikuti data, bukan seluruh rentang instrumen. |
| `evaluasi_silhouette_kmeans_177.png` | Koefisien silhouette seluruh 177 assignment; garis putus-putus menunjukkan rata-ratanya. Nilai data latih dicantumkan terpisah. |
| `struktur_pohon_keputusan_asli_rf_177.png` | Cuplikan Tree #1 hingga kedalaman 3. Nomor lingkaran adalah nomor node asli. Penanda cabang berlanjut berarti node tersebut belum merupakan daun akhir. |
| `visualisasi_pohon_keputusan_rf_177.png` | Tree #1 penuh, berisi 51 node dan 26 daun dari estimator aktual. Gunakan SVG atau zoom untuk membaca seluruh aturan. |
| `diagram_arsitektur_random_forest.png` | Ensemble 100 pohon: empat cuplikan pohon, suara tiap pohon, majority voting, dan final class untuk satu siswa contoh. |
| `diagram_pohon_lingkaran_rf.png` / `.jpg` | Versi diagram ensemble dengan nomor node pada simpul lingkaran. |
| `diagram_pohon_final_rf.png` / `.jpg` | Salinan diagram ensemble utama dengan nama berkas yang pernah digunakan sebelumnya. |
| `diagram_alur_hybrid_177.png` | Alur seluruh sistem: input → persiapan fitur → K-Means → RF → hasil. Berbeda tujuan dengan diagram ensemble RF. |

PNG dapat dimasukkan ke Word. SVG adalah gambar vektor yang dapat diperbesar tanpa teks pecah. Gambar ini dapat direproduksi melalui fungsi plot pada `research_pipeline.py`, yang memanggil `research_figures.py` dan membaca bundle tersimpan.

## Versi pada metadata penelitian

Nomor versi bundle model NALARA disimpan pada metadata dan dokumentasi untuk membedakan hasil revisi dari bundle sebelumnya; nomor tersebut tidak ditampilkan pada gambar laporan. Versi 1.1.0 menggunakan scaler, K-Means, dan Random Forest yang dipelajari dari 141 data latih, dengan 36 data uji terpisah. Nomor versi bukan nilai akurasi, jumlah klaster, atau parameter algoritma. Perbaikan tata letak gambar tidak menghasilkan versi model baru karena estimator dan hasil numeriknya tetap sama.

Kalimat yang dapat digunakan pada laporan: "Pemodelan menggunakan bundle NALARA versi 1.1.0, dengan StandardScaler, K-Means, dan Random Forest dilatih pada 141 siswa serta evaluasi holdout pada 36 siswa."

## Rekomendasi pemakaian dalam laporan

Untuk ilustrasi mekanisme RF, gunakan dua gambar berikut pada Bab IV:

1. `diagram_pohon_lingkaran_rf.png`: menjelaskan kumpulan 100 pohon, agregasi suara, dan final class, dengan nomor node pada simpul.
2. `struktur_pohon_keputusan_asli_rf_177.png`: menunjukkan contoh aturan dan percabangan aktual di dalam Tree #1. Caption harus menyebutkan bahwa gambar merupakan cuplikan sampai kedalaman 3.

Gambar pohon penuh `visualisasi_pohon_keputusan_rf_177.png` tersedia lokal sebagai alternatif untuk halaman landscape atau lampiran. Pohon penuh memuat seluruh 51 node dan 26 daun, sedangkan cuplikan menunjukkan kedalaman 0 sampai 3. Keduanya adalah estimator yang sama. Label fitur/daun pada pohon penuh disingkat agar tidak bertumpuk, dan dijelaskan melalui keterangan serta legenda. Diagram arsitektur tanpa nomor dan `diagram_pohon_final_rf` merupakan variasi tampilan ensemble; tidak perlu memasukkannya sekaligus.

Pada Bab V, gunakan `confusion_matrix_rf_177.png` untuk hasil klasifikasi, `feature_importance_rf_177.png` untuk kepentingan fitur, serta `hasil_voting_nyata_100_pohon_177.png` jika membahas rekap suara dan confidence seluruh 36 siswa uji. Diagram `diagram_alur_hybrid_177.png` dapat digunakan untuk memperkenalkan hubungan tahap K-Means dan RF sebelum pembahasan kedua algoritma.

Pohon keputusan menggambarkan aturan internal **satu** estimator. Arsitektur RF menggambarkan hubungan **banyak** estimator dan cara kontribusinya digabung menjadi prediksi akhir. Alur hybrid menggambarkan seluruh proses sistem, termasuk persiapan data dan K-Means.

## Majority voting dan final class

Satu pohon keputusan hanya memberikan satu kontribusi; agregasi dan final class berada pada diagram **ensemble**, bukan pada setiap cabang satu pohon.

Implementasi `RandomForestClassifier` memilih kelas dengan rata-rata probabilitas tertinggi dari seluruh pohon. Hard voting menghitung kelas yang dipilih masing-masing pohon. Keduanya tidak selalu sama pada semua model RF.

Pada bundle NALARA 1.1.0, seluruh 2.868 daun dari 100 pohon murni satu kelas. Karena probabilitas daun merupakan 0 atau 1, rata-rata probabilitas sama dengan proporsi suara. Pemeriksaan 177 profil menghasilkan selisih maksimum 0 dan kelas pemenang sama. Dengan kondisi bundle ini, majority voting dan final class dapat ditampilkan tanpa mengubah algoritma prediksi.

Contoh aktual pada diagram: `Siswa_018`, jurusan RPL. Voting: Bisnis 18, Pariwisata 2, Sains & IT 75, Seni & Desain 5, Sosial 0. Final class: Sains & Teknologi Informasi, confidence 75%. Kode K-Means siswa adalah 0/Tidak Sesuai, tetapi hubungan hasil dengan jurusan asal tetap Linier. Ini menunjukkan bahwa kode klaster merupakan fitur tambahan, bukan saklar yang memaksa rekomendasi lintas bidang.

Confidence adalah skor keluaran model, bukan peluang keberhasilan kuliah. Prediksi penelitian dievaluasi terhadap label proksi berbasis aturan.

## Apa arti bootstrap?

Bootstrap pada Random Forest adalah pengambilan sampel data latih secara acak **dengan pengembalian** untuk membentuk kumpulan pelatihan tiap pohon. Satu siswa dapat terambil berulang, sedangkan siswa lain tidak terambil oleh pohon tertentu. Pemilihan kandidat fitur secara acak juga berlangsung saat pembentukan percabangan.

Proses itu berlangsung saat **pelatihan**, bukan setiap kali siswa baru meminta prediksi. Saat inferensi, sistem menggunakan 100 pohon yang sudah tersimpan.

Karena ada pengambilan berulang, jumlah sampel unik pada node dapat berbeda dari jumlah berbobot. Tree #1, misalnya, memiliki 94 sampel unik pada akar dengan total bobot 141. Detail aturan, Gini, jumlah unik, dan bobot per kelas tersedia di `detail_node_tree_1_177.csv`. Kata bootstrap tidak perlu diulang pada setiap judul gambar; penjelasan metodologinya dapat ditempatkan di uraian pelatihan RF.

## Mengapa silhouette menyebut 141 data latih?

Model dan centroid dipelajari dari 141 siswa. Model yang sama kemudian memberikan assignment pada seluruh 177 siswa. Grafik silhouette menggunakan fitur Z-score dan assignment 177 siswa, dengan rata-rata 0,330059. Silhouette khusus 141 siswa latih adalah 0,340531.

Keterangan tersebut menjelaskan cakupan angka dan mencegah pembaca mengira centroid dilatih menggunakan data uji. Keterangan tetap disediakan pada catatan gambar, dengan judul yang lebih ringkas. Silhouette dihitung dari jarak antarsampel dan kelompoknya; tidak dihitung langsung dari jarak ke centroid.

## Mengapa scatter sebelumnya menumpuk di kanan atas?

Sumbu sebelumnya mencakup seluruh rentang X 7–35 dan Y 0–100. Sementara itu, data aktual memiliki X 18–33 dan Y 66,25–92,25. Akibatnya banyak ruang kosong dan titik terlihat kecil di kanan atas.

Versi revisi memakai batas tampilan dekat nilai minimum/maksimum dengan margin. Semua 177 titik tetap digambar, termasuk nilai rapor terendah; koordinat dan centroid tidak diubah, serta tidak ditambahkan jitter. Titik dengan koordinat sama tetap dapat bertumpuk. Rentang instrumen asli tetap dijelaskan pada catatan gambar.

Grafik memakai satuan mentah agar mudah dibaca. Perhitungan keanggotaan klaster tetap menggunakan jarak di ruang Z-score, sehingga penilaian jarak tidak dilakukan dengan penggaris pada gambar bersumbu mentah.

Rujukan implementasi: [RandomForestClassifier — bootstrap dan predict_proba](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [analisis silhouette](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html).
