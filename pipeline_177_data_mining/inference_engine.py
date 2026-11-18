# ==============================================================================
# PROYEK DATA MINING NALARA - INFERENCE ENGINE
# File: pipeline_177_data_mining/inference_engine.py
# Engine Inferensi Prediksi Jalur Rekomendasi Rumpun Program Studi
# ==============================================================================

import os
import re
import math
import joblib
import numpy as np

class NalaraInferenceEngine:
    # 5 Kode Jurusan Kanonik SMKN 1 Bintan Timur
    CANONICAL_JURUSAN = {'RPL', 'AKL', 'PH', 'DKV', 'TB'}

    # Daftar Alias Lengkap & Eksplisit (Pencocokan Kata Utuh / Whitelist Ketat)
    JURUSAN_ALIAS_MAP = {
        # RPL -> Investigative
        'RPL': 'RPL',
        'REKAYASA PERANGKAT LUNAK': 'RPL',
        'TEKNIK REKAYASA PERANGKAT LUNAK': 'RPL',
        'REKAYASA PERANGKAT LUNAK (RPL)': 'RPL',
        'REKAYASA PERANGKAT LUNAK RPL': 'RPL',
        
        # AKL -> Conventional
        'AKL': 'AKL',
        'AKT': 'AKL',
        'AK': 'AKL',
        'AKUNTANSI': 'AKL',
        'AKUNTANSI DAN KEUANGAN LEMBAGA': 'AKL',
        'AKUNTANSI KEUANGAN LEMBAGA': 'AKL',
        'AKUNTANSI (AKT)': 'AKL',
        'AKUNTANSI (AKL)': 'AKL',
        'AKUNTANSI AKT': 'AKL',
        'AKUNTANSI AKL': 'AKL',
        
        # PH -> Social
        'PH': 'PH',
        'PERHOTELAN': 'PH',
        'JASA PERHOTELAN': 'PH',
        'PERHOTELAN (PH)': 'PH',
        'PERHOTELAN PH': 'PH',
        
        # DKV -> Artistic
        'DKV': 'DKV',
        'DESAIN KOMUNIKASI VISUAL': 'DKV',
        'DESAIN KOMUNIKASI VISUAL (DKV)': 'DKV',
        'DESAIN KOMUNIKASI VISUAL DKV': 'DKV',
        
        # TB -> Artistic
        'TB': 'TB',
        'TATA BUSANA': 'TB',
        'BUSANA': 'TB',
        'DESAIN DAN PRODUKSI BUSANA': 'TB',
        'TATA BUSANA (TB)': 'TB',
        'TATA BUSANA TB': 'TB'
    }

    # Pemetaan Jurusan Kanonik ke Dimensi Minat Kejuruan Spesifik (K-Means X-Axis)
    JURUSAN_TO_RIASEC_DIM = {
        'RPL': 'Investigative',
        'AKL': 'Conventional',
        'PH': 'Social',
        'DKV': 'Artistic',
        'TB': 'Artistic'
    }

    RIASEC_DIMENSIONS = [
        'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional'
    ]

    # Pemetaan 42 Butir Kuesioner ke 6 Dimensi Holland (7 butir per dimensi: 5 Minat + 2 Efikasi Diri)
    # Berdasarkan struktur instrumen bab3_sub1_data_cleaning_177.py
    INDEX_42_TO_RIASEC = {
        'Realistic': [0, 6, 12, 18, 24, 30, 36],
        'Investigative': [1, 7, 13, 19, 25, 31, 37],
        'Artistic': [2, 8, 14, 20, 26, 32, 38],
        'Social': [3, 9, 15, 21, 27, 33, 39],
        'Enterprising': [4, 10, 16, 22, 28, 34, 40],
        'Conventional': [5, 11, 17, 23, 29, 35, 41]
    }

    MODEL_METADATA = {
        'model_version': '1.0.0',
        'pipeline_version': 'pipeline_177_smkn1_bintan_timur',
        'algorithm': 'Two-Stage Hybrid: K-Means 2D (K=2) + Random Forest Classifier (100 Trees)',
        'target_sample_origin': 'SMKN 1 Bintan Timur (177 Siswa)',
        'features_count': 11
    }

    def __init__(self, bundle_path=None):
        if bundle_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            bundle_path = os.path.join(base_dir, "output_model", "pipeline_bundle_177.joblib")
            
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Model bundle not found at: {bundle_path}")
            
        self.bundle = joblib.load(bundle_path)
        self.scaler = self.bundle['scaler']
        self.fitur_numerik_scaler = self.bundle['fitur_numerik_scaler']
        self.kmeans = self.bundle['kmeans']
        self.invert_clusters = self.bundle['invert_clusters']
        self.rf_model = self.bundle['rf_model']
        self.features_rf = self.bundle['features_rf']
        self.target_classes = list(self.bundle['target_classes'])

    @classmethod
    def normalize_jurusan(cls, jurusan_raw):
        """
        Normalisasi jurusan berbasis whitelist ketat & alias kanonik eksplisit.
        Mencegah penerimaan kata parsial acak (misal: 'REKAYASA KIMIA', 'DESAIN INTERIOR', 'DKV TB').
        """
        if not isinstance(jurusan_raw, str) or not jurusan_raw.strip():
            return None
            
        clean_str = " ".join(jurusan_raw.strip().upper().split())
        if clean_str in cls.JURUSAN_ALIAS_MAP:
            return cls.JURUSAN_ALIAS_MAP[clean_str]
            
        # Coba pembersihan tanda kurung dan karakter non-alfanumerik
        clean_punct = re.sub(r'[^\w\s]', '', clean_str)
        clean_punct = " ".join(clean_punct.split())
        if clean_punct in cls.JURUSAN_ALIAS_MAP:
            return cls.JURUSAN_ALIAS_MAP[clean_punct]
            
        return None

    @classmethod
    def aggregate_raw_42_answers(cls, raw_answers):
        """
        Mengonversi 42 jawaban mentah Likert (skala 1-5) menjadi 6 skor komposit RIASEC (skala 7-35).
        raw_answers dapat berupa list 42 integer atau dict dengan kunci tepat 0..41 atau 1..42.
        Nomor kunci di luar himpunan kanonik tersebut (misal 100..141) wajib ditolak.
        """
        VALID_KEYS_0_INDEXED = {i for i in range(42)}
        VALID_KEYS_1_INDEXED = {i for i in range(1, 43)}

        if isinstance(raw_answers, dict):
            try:
                int_keys = {int(k) for k in raw_answers.keys()}
            except (ValueError, TypeError):
                raise ValueError("Kunci dictionary jawaban mentah harus berupa angka butir pertanyaan.")

            if int_keys == VALID_KEYS_0_INDEXED:
                answers_list = [raw_answers[i] if i in raw_answers else raw_answers[str(i)] for i in range(42)]
            elif int_keys == VALID_KEYS_1_INDEXED:
                answers_list = [raw_answers[i] if i in raw_answers else raw_answers[str(i)] for i in range(1, 43)]
            else:
                raise ValueError(
                    f"Himpunan nomor butir kuesioner tidak valid (diterima rentang kunci: {min(int_keys)} s/d {max(int_keys)}). "
                    "Nomor butir kuesioner wajib tepat 0..41 (0-indexed) atau 1..42 (1-indexed)."
                )
        elif isinstance(raw_answers, (list, tuple)):
            if len(raw_answers) != 42:
                raise ValueError(f"Daftar jawaban mentah harus berisi tepat 42 butir (diterima: {len(raw_answers)} butir).")
            answers_list = list(raw_answers)
        else:
            raise ValueError("Input jawaban mentah harus berupa list 42 butir atau dictionary 42 butir.")

        # Validasi setiap butir adalah integer 1-5
        clean_answers = []
        for idx, val in enumerate(answers_list):
            if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
                raise ValueError(f"Jawaban butir ke-{idx+1} harus berupa angka 1-5 (tipe boolean tidak diizinkan; diterima: {val}).")
            f_val = float(val)
            if not f_val.is_integer() or not (1 <= int(round(f_val)) <= 5):
                raise ValueError(f"Jawaban butir ke-{idx+1} ({val}) harus berupa bilangan bulat skala Likert 1 hingga 5.")
            clean_answers.append(int(round(f_val)))

        # Agregasi penjumlahan 7 butir per dimensi sesuai instrumen penelitian
        composite_scores = {}
        for dim, item_indices in cls.INDEX_42_TO_RIASEC.items():
            composite_scores[dim] = sum(clean_answers[i] for i in item_indices)
            
        return composite_scores

    def validate_and_normalize_inputs(self, jurusan, matematika, bahasa_indonesia, bahasa_inggris, produktif, riasec_input):
        """
        Validasi ketat tipe data, batas nilai (range), ketiadaan nilai NaN/Inf/Null,
        pencegahan boolean coercion, integer check pada skor RIASEC komposit,
        serta pemetaan alias jurusan secara presisi (non-substring sembarang).
        Mendukung riasec_input berupa dict 6 dimensi (skala 7-35) ATAU list 42 jawaban (skala 1-5).
        """
        # 1. Validasi Jurusan (Pencocokan Kanonik Eksplisit)
        canon_jurusan = self.normalize_jurusan(jurusan)
        if canon_jurusan is None:
            raise ValueError(
                f"Jurusan '{jurusan}' tidak dikenali. Pilihan yang didukung: "
                "RPL (Rekayasa Perangkat Lunak), AKL (Akuntansi), PH (Perhotelan), "
                "DKV (Desain Komunikasi Visual), TB (Tata Busana)."
            )
        dim_minat = self.JURUSAN_TO_RIASEC_DIM[canon_jurusan]

        # 2. Validasi Nilai Rapor (Menolak tipe boolean, NaN, Inf, atau di luar 0-100)
        nilai_dict = {
            'Matematika': matematika,
            'Bahasa_Indonesia': bahasa_indonesia,
            'Bahasa_Inggris': bahasa_inggris,
            'Produktif': produktif
        }
        
        clean_nilai = {}
        for mapel, val in nilai_dict.items():
            if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
                raise ValueError(f"Nilai '{mapel}' harus berupa angka numerik valid (tipe boolean tidak diizinkan; diterima: {val}).")
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                raise ValueError(f"Nilai '{mapel}' tidak boleh bernilai NaN atau Tak Hingga (Inf).")
            if not (0.0 <= f_val <= 100.0):
                raise ValueError(f"Nilai '{mapel}' berada di luar rentang valid [0.0 - 100.0] (diterima: {f_val}).")
            clean_nilai[mapel] = f_val

        # 3. Validasi & Penanganan Masukan RIASEC (Bisa berupa 6 skor atau 42 jawaban)
        if isinstance(riasec_input, (list, tuple)) or (isinstance(riasec_input, dict) and len(riasec_input) == 42 and all(isinstance(k, int) or str(k).isdigit() for k in riasec_input.keys())):
            clean_riasec_scores = self.aggregate_raw_42_answers(riasec_input)
        elif isinstance(riasec_input, dict):
            clean_riasec_scores = {}
            for dim in self.RIASEC_DIMENSIONS:
                if dim not in riasec_input:
                    raise ValueError(f"Dimensi RIASEC '{dim}' wajib disertakan.")
                val_r = riasec_input[dim]
                if isinstance(val_r, bool) or not isinstance(val_r, (int, float, np.number)):
                    raise ValueError(f"Skor RIASEC '{dim}' harus berupa angka numerik (tipe boolean tidak diizinkan; diterima: {val_r}).")
                f_r = float(val_r)
                if math.isnan(f_r) or math.isinf(f_r):
                    raise ValueError(f"Skor RIASEC '{dim}' tidak boleh bernilai NaN atau Tak Hingga.")
                if not f_r.is_integer():
                    raise ValueError(
                        f"Skor RIASEC '{dim}' ({val_r}) harus berupa bilangan bulat integer "
                        "(hasil penjumlahan 7 butir Likert skala 1-5 menghasilkan bilangan bulat 7 hingga 35)."
                    )
                int_r = int(round(f_r))
                if not (7 <= int_r <= 35):
                    raise ValueError(
                        f"Skor RIASEC '{dim}' ({int_r}) berada di luar rentang instrumen [7 - 35] "
                        "(gabungan 7 butir Likert skala 1-5)."
                    )
                clean_riasec_scores[dim] = float(int_r)
        else:
            raise ValueError("Input RIASEC harus berupa dictionary 6 dimensi (skor 7-35) atau list/dict 42 butir jawaban (skala 1-5).")

        return canon_jurusan, clean_nilai, clean_riasec_scores, dim_minat

    def predict_student(self, jurusan, matematika, bahasa_indonesia, bahasa_inggris, produktif, riasec_scores):
        """
        Melakukan inferensi lengkap untuk satu profil siswa dengan validasi ketat,
        pembagian voting 100 pohon yang tepat, dan vektor probabilitas penuh.
        """
        canon_jurusan, clean_nilai, clean_riasec, dim_minat = self.validate_and_normalize_inputs(
            jurusan, matematika, bahasa_indonesia, bahasa_inggris, produktif, riasec_scores
        )

        mat = clean_nilai['Matematika']
        bi = clean_nilai['Bahasa_Indonesia']
        en = clean_nilai['Bahasa_Inggris']
        prod = clean_nilai['Produktif']
        rata_rapor = (mat + bi + en + prod) / 4.0
        skor_minat = clean_riasec[dim_minat]

        # 1. Transformasi 12 Fitur Numerik dengan StandardScaler Presisi Penuh
        raw_row = [
            mat, bi, en, prod, rata_rapor,
            clean_riasec['Realistic'], clean_riasec['Investigative'], clean_riasec['Artistic'],
            clean_riasec['Social'], clean_riasec['Enterprising'], clean_riasec['Conventional'],
            skor_minat
        ]
        
        scaled_vals = self.scaler.transform([raw_row])[0]
        scaled_dict = dict(zip([f"{c}_scaled" for c in self.fitur_numerik_scaler], scaled_vals))

        # 2. Diagnosa K-Means 2D & Perhitungan Jarak Euclidean Sebenarnya
        X_kmeans = np.array([[scaled_dict['Skor_Minat_Kejuruan_scaled'], scaled_dict['Rata_Rapor_scaled']]])
        raw_cluster = int(self.kmeans.predict(X_kmeans)[0])
        klaster_kmeans = int(1 - raw_cluster) if self.invert_clusters else raw_cluster
        
        # Jarak Euclidean nyata dari titik siswa ke pusat centroid terdekat
        centroid_terdekat = self.kmeans.cluster_centers_[raw_cluster]
        jarak_euclidean_real = float(np.linalg.norm(X_kmeans[0] - centroid_terdekat))

        # 3. Prediksi Stage 2: Random Forest (10 fitur Z-Score + 1 fitur Klaster biner 0/1)
        X_rf = np.array([[
            scaled_dict['Matematika_scaled'],
            scaled_dict['Bahasa_Indonesia_scaled'],
            scaled_dict['Bahasa_Inggris_scaled'],
            scaled_dict['Produktif_scaled'],
            scaled_dict['Realistic_scaled'],
            scaled_dict['Investigative_scaled'],
            scaled_dict['Artistic_scaled'],
            scaled_dict['Social_scaled'],
            scaled_dict['Enterprising_scaled'],
            scaled_dict['Conventional_scaled'],
            klaster_kmeans
        ]])

        predicted_class = str(self.rf_model.predict(X_rf)[0])
        proba_array = self.rf_model.predict_proba(X_rf)[0]
        proba_dict = {cls: float(prob) for cls, prob in zip(self.target_classes, proba_array)}

        # 4. Agregasi Voting 100 Pohon Individual dengan Pemetaan Indeks Eksak
        vote_counts = {cls: 0 for cls in self.target_classes}
        for tree in self.rf_model.estimators_:
            pred_leaf = tree.predict(X_rf)[0]
            cls_idx = int(round(float(pred_leaf)))
            cls_name = self.target_classes[cls_idx]
            vote_counts[cls_name] += 1

        total_votes = sum(vote_counts.values())
        if total_votes != len(self.rf_model.estimators_):
            raise RuntimeError(f"Total voting pohon ({total_votes}) tidak sama dengan jumlah estimator ({len(self.rf_model.estimators_)}).")

        # 5. Narasi Profil Minat-Akademik Berdasarkan Karakteristik Kelompok Centroid Terukur
        if klaster_kmeans == 1:
            profil_klaster = {
                'kode': 1,
                'label': 'Klaster 1 (Kecenderungan Minat Kejuruan Tinggi)',
                'karakteristik_kelompok': (
                    'Siswa dalam kelompok ini memiliki skor minat kejuruan pada bidang SMK asalnya '
                    'yang berada di atas rata-rata sampel (Centroid Minat: 28.46, Rapor: 83.39).'
                ),
                'catatan_konseling': (
                    'Orientasi minat kejuruan yang tinggi pada bidang studi saat ini dapat menjadi '
                    'pertimbangan untuk memprioritaskan program studi yang linier.'
                )
            }
        else:
            profil_klaster = {
                'kode': 0,
                'label': 'Klaster 0 (Kecenderungan Minat Kejuruan Lebih Rendah / Nilai Rapor Kompetitif)',
                'karakteristik_kelompok': (
                    'Siswa dalam kelompok ini memiliki skor minat kejuruan pada bidang SMK asalnya '
                    'yang berada di bawah rata-rata sampel (Centroid Minat: 24.23), dengan rata-rata '
                    'nilai rapor yang kompetitif (Centroid Rapor: 84.47).'
                ),
                'catatan_konseling': (
                    'Perbedaan antara skor minat pada kejuruan saat ini dan capaian akademik menjadi '
                    'bahan diskusi bersama Guru BK untuk menelaah kesesuaian arah kelanjutan studi.'
                )
            }

        return {
            'jurusan_kanonik': canon_jurusan,
            'rumpun_terprediksi': predicted_class,
            'confidence_score_pct': round(float(np.max(proba_array) * 100), 2),
            'probabilitas_per_rumpun': proba_dict,
            'voting_pohon': vote_counts,
            'total_pohon': total_votes,
            'klaster_kmeans': profil_klaster,
            'jarak_euclidean': round(jarak_euclidean_real, 4),
            'skor_riasec_komposit': {k: int(v) for k, v in clean_riasec.items()},
            'fitur_terstandarisasi': scaled_dict,
            'model_metadata': self.MODEL_METADATA
        }

if __name__ == '__main__':
    engine = NalaraInferenceEngine()
    print("[V] NalaraInferenceEngine v1.3 berhasil dimuat!")
    print(f"    - Target Classes ({len(engine.target_classes)}): {engine.target_classes}")
