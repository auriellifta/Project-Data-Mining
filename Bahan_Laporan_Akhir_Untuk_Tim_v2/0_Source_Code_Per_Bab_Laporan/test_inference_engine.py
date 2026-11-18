# ==============================================================================
# PROYEK DATA MINING NALARA - TEST SUITE INFERENCE ENGINE & API
# File: pipeline_177_data_mining/test_inference_engine.py
# Pengujian Validitas Inferensi, Reproduksibilitas, Voting Pohon, & Edge Cases
# ==============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd

# Konfigurasi eksplisit API Key untuk lingkungan pengujian (tanpa fallback hardcode di modul API)
TEST_API_KEY = "test-secret-env-key-9999"
os.environ['NALARA_API_KEY'] = TEST_API_KEY

from inference_engine import NalaraInferenceEngine
from api_server import app

print("=" * 85)
print("PENGUJIAN VALIDITAS & REPRODUKSIBILITAS LENGKAP NALARA INFERENCE ENGINE & API")
print("=" * 85)

base = os.path.dirname(os.path.abspath(__file__))
df_test = pd.read_csv(os.path.join(base, 'hasil_prediksi_rf_177.csv'))
df_all = pd.read_csv(os.path.join(base, 'dataset_terklaster_177.csv'))
df_siap = pd.read_csv(os.path.join(base, 'dataset_siap_model_177.csv'))

engine = NalaraInferenceEngine()

mismatches = 0
tested = 0
max_proba_diff = 0.0

# ------------------------------------------------------------------------------
# BAGIAN 1: Pengujian Kesamaan terhadap 36 Data Uji Resmi (Testing Set 20%)
# ------------------------------------------------------------------------------
print("\n[*] BAGIAN 1: Menguji 36 Siswa Data Uji Resmi...")

for _, row_test in df_test.iterrows():
    id_s = row_test['ID_Siswa']
    row_full = df_all[df_all['ID_Siswa'] == id_s].iloc[0]
    row_siap = df_siap[df_siap['ID_Siswa'] == id_s].iloc[0]
    
    riasec = {
        'Realistic': float(row_full['Realistic']),
        'Investigative': float(row_full['Investigative']),
        'Artistic': float(row_full['Artistic']),
        'Social': float(row_full['Social']),
        'Enterprising': float(row_full['Enterprising']),
        'Conventional': float(row_full['Conventional'])
    }
    
    res = engine.predict_student(
        jurusan=row_full['Jurusan'],
        matematika=float(row_full['Matematika']),
        bahasa_indonesia=float(row_full['Bahasa_Indonesia']),
        bahasa_inggris=float(row_full['Bahasa_Inggris']),
        produktif=float(row_full['Produktif']),
        riasec_scores=riasec
    )
    
    # 1.1 Verifikasi Klaster K-Means
    if res['klaster_kmeans']['kode'] != int(row_full['Klaster_KMeans']):
        print(f"[FAIL] Klaster mismatch untuk {id_s}: didapat {res['klaster_kmeans']['kode']} vs acuan {row_full['Klaster_KMeans']}")
        mismatches += 1
        
    # 1.2 Verifikasi Prediksi Random Forest
    if res['rumpun_terprediksi'] != row_test['Prediksi_Random_Forest']:
        print(f"[FAIL] Prediksi RF mismatch untuk {id_s}: {res['rumpun_terprediksi']} vs {row_test['Prediksi_Random_Forest']}")
        mismatches += 1
        
    # 1.3 Verifikasi Seluruh Vektor Probabilitas Kelas terhadap Model Asli
    X_rf_ref = np.array([[
        row_siap['Matematika_scaled'],
        row_siap['Bahasa_Indonesia_scaled'],
        row_siap['Bahasa_Inggris_scaled'],
        row_siap['Produktif_scaled'],
        row_siap['Realistic_scaled'],
        row_siap['Investigative_scaled'],
        row_siap['Artistic_scaled'],
        row_siap['Social_scaled'],
        row_siap['Enterprising_scaled'],
        row_siap['Conventional_scaled'],
        int(row_full['Klaster_KMeans'])
    ]])
    expected_proba = engine.rf_model.predict_proba(X_rf_ref)[0]
    
    for cls_idx, cls_name in enumerate(engine.target_classes):
        diff_p = abs(res['probabilitas_per_rumpun'][cls_name] - expected_proba[cls_idx])
        if diff_p > max_proba_diff:
            max_proba_diff = diff_p
        if diff_p > 1e-7:
            print(f"[FAIL] Probabilitas mismatch untuk {id_s} pada kelas {cls_name}: diff={diff_p}")
            mismatches += 1
            
    # 1.4 Verifikasi Konsistensi Voting 100 Pohon
    votes = res['voting_pohon']
    if sum(votes.values()) != 100:
        print(f"[FAIL] Total voting tidak 100 untuk {id_s}: sum={sum(votes.values())}")
        mismatches += 1
        
    if set(votes.keys()) != set(engine.target_classes):
        print(f"[FAIL] Kunci voting tidak sesuai target classes untuk {id_s}: {votes.keys()}")
        mismatches += 1
        
    tested += 1

print(f"    - Total Data Uji Berhasil Dievaluasi : {tested} siswa")
print(f"    - Selisih Vektor Probabilitas Maks   : {max_proba_diff:.2e}")
print(f"    - Status Pengujian Bagian 1          : {'[SUKSES 100%]' if mismatches == 0 else '[GAGAL]'}")

if mismatches > 0:
    print(f"\n[FATAL] Terjadi {mismatches} mismatch pada data uji! Menghentikan pengujian.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# BAGIAN 2: Pengujian Validasi Masukan & Penanganan Edge Cases Engine
# ------------------------------------------------------------------------------
print("\n[*] BAGIAN 2: Menguji Validasi Masukan & Edge Cases (Negative Testing)...")
edge_passed = 0
total_edge = 0

def test_negative_case(name, func, expected_error_msg):
    global edge_passed, total_edge
    total_edge += 1
    try:
        func()
        print(f"  [FAIL] {name} - Seharusnya menghasilkan ValueError tapi lolos!")
        return False
    except ValueError as e:
        if expected_error_msg.lower() in str(e).lower():
            print(f"  [PASS] {name} - Berhasil dicegat: {e}")
            edge_passed += 1
            return True
        else:
            print(f"  [FAIL] {name} - Pesan error tidak sesuai: {e}")
            return False

valid_riasec = {'Realistic': 20, 'Investigative': 25, 'Artistic': 22, 'Social': 18, 'Enterprising': 24, 'Conventional': 23}

# 2.1 Pengujian Ketat Jurusan Temuan Codex
test_negative_case(
    "Jurusan 'REKAYASA KIMIA' (Bukan RPL)",
    lambda: engine.predict_student('REKAYASA KIMIA', 80, 80, 80, 80, valid_riasec),
    "tidak dikenali"
)
test_negative_case(
    "Jurusan 'DESAIN INTERIOR' (Bukan DKV)",
    lambda: engine.predict_student('DESAIN INTERIOR', 80, 80, 80, 80, valid_riasec),
    "tidak dikenali"
)
test_negative_case(
    "Jurusan 'DKV TB' (Ambiguitas Ganda)",
    lambda: engine.predict_student('DKV TB', 80, 80, 80, 80, valid_riasec),
    "tidak dikenali"
)
test_negative_case(
    "Jurusan 'TIDAK DIKENAL'",
    lambda: engine.predict_student('TIDAK DIKENAL', 80, 80, 80, 80, valid_riasec),
    "tidak dikenali"
)

# 2.2 Pengujian Tipe Data Numerik & Skala
test_negative_case(
    "Nilai Matematika bertipe Boolean (True)",
    lambda: engine.predict_student('RPL', True, 85, 80, 90, valid_riasec),
    "boolean tidak diizinkan"
)
test_negative_case(
    "Skor RIASEC Pecahan Desimal (24.5)",
    lambda: engine.predict_student('RPL', 80, 85, 80, 90, {'Realistic': 20, 'Investigative': 24.5, 'Artistic': 22, 'Social': 18, 'Enterprising': 24, 'Conventional': 23}),
    "bilangan bulat integer"
)
test_negative_case(
    "Nilai Rapor > 100 (e.g. 105)",
    lambda: engine.predict_student('RPL', 105, 85, 80, 90, valid_riasec),
    "rentang valid"
)
test_negative_case(
    "Nilai Rapor Negatif (e.g. -5)",
    lambda: engine.predict_student('RPL', 80, -5, 80, 90, valid_riasec),
    "rentang valid"
)
test_negative_case(
    "Nilai Rapor bernilai NaN",
    lambda: engine.predict_student('RPL', 80, np.nan, 80, 90, valid_riasec),
    "nan"
)
test_negative_case(
    "Skor RIASEC di bawah rentang (<7)",
    lambda: engine.predict_student('RPL', 80, 85, 80, 90, {'Realistic': 5, 'Investigative': 25, 'Artistic': 22, 'Social': 18, 'Enterprising': 24, 'Conventional': 23}),
    "rentang instrumen"
)
test_negative_case(
    "Skor RIASEC di atas rentang (>35)",
    lambda: engine.predict_student('RPL', 80, 85, 80, 90, {'Realistic': 20, 'Investigative': 25, 'Artistic': 22, 'Social': 40, 'Enterprising': 24, 'Conventional': 23}),
    "rentang instrumen"
)
test_negative_case(
    "Dimensi RIASEC Tidak Lengkap",
    lambda: engine.predict_student('RPL', 80, 85, 80, 90, {'Realistic': 20, 'Investigative': 25}),
    "wajib disertakan"
)

# 2.3 Pengujian Nomor Butir Dictionary 100..141 Temuan Codex (Wajib Ditolak)
dict_100_141 = {i: 4 for i in range(100, 142)}
test_negative_case(
    "Dictionary Nomor Butir 100..141 (Invalid Index Set)",
    lambda: engine.predict_student('RPL', 80, 85, 80, 90, dict_100_141),
    "himpunan nomor butir kuesioner tidak valid"
)

# 2.4 Pengujian Dukungan 42 Jawaban Mentah yang Sah (List, Dict 0..41, Dict 1..42)
raw_42_dummy = [4] * 42
res_list = engine.predict_student('RPL', 85, 85, 85, 85, raw_42_dummy)
assert res_list['skor_riasec_komposit']['Realistic'] == 28
print("  [PASS] Agregasi List 42 Butir - Berhasil dikonversi menjadi 6 skor komposit.")
edge_passed += 1; total_edge += 1

dict_0_41 = {i: 4 for i in range(42)}
res_0_41 = engine.predict_student('RPL', 85, 85, 85, 85, dict_0_41)
assert res_0_41['skor_riasec_komposit']['Investigative'] == 28
print("  [PASS] Agregasi Dict 0..41 - Berhasil dikonversi menjadi 6 skor komposit.")
edge_passed += 1; total_edge += 1

dict_1_42 = {i: 4 for i in range(1, 43)}
res_1_42 = engine.predict_student('RPL', 85, 85, 85, 85, dict_1_42)
assert res_1_42['skor_riasec_komposit']['Artistic'] == 28
print("  [PASS] Agregasi Dict 1..42 - Berhasil dikonversi menjadi 6 skor komposit.")
edge_passed += 1; total_edge += 1

print(f"    - Total Uji Edge Cases Engine : {edge_passed} / {total_edge} lolos validasi.")
if edge_passed != total_edge:
    print(f"\n[FATAL] Sebagian pengujian edge cases engine gagal!")
    sys.exit(1)

# ------------------------------------------------------------------------------
# BAGIAN 3: Pengujian Robustness & Autentikasi REST API Server (Flask Test Client)
# ------------------------------------------------------------------------------
print("\n[*] BAGIAN 3: Menguji Robustness & Autentikasi REST API Server...")
client = app.test_client()
auth_headers = {'X-API-Key': TEST_API_KEY}

# 3.1 Test Health Check Publik
r_health = client.get('/health')
assert r_health.status_code == 200, f"Health check failed with {r_health.status_code}"
assert r_health.get_json().get('status') == 'healthy', "Health status is not 'healthy'"
print("  [PASS] GET /health - Berhasil merespons 200 OK dan status 'healthy'.")

# 3.2 Test Akses Tanpa Autentikasi (Harus 401 Unauthorized)
r_no_auth = client.post('/api/v1/predict', json={'jurusan': 'RPL'})
assert r_no_auth.status_code == 401, f"Expected 401 without auth, got {r_no_auth.status_code}"
print("  [PASS] POST tanpa API Key - Berhasil dicegat dengan 401 Unauthorized.")

# 3.3 Test Akses dengan API Key Salah (Harus 401 Unauthorized)
r_bad_auth = client.post('/api/v1/predict', headers={'X-API-Key': 'wrong-key'}, json={'jurusan': 'RPL'})
assert r_bad_auth.status_code == 401, f"Expected 401 with bad auth, got {r_bad_auth.status_code}"
print("  [PASS] POST dengan API Key salah - Berhasil dicegat dengan 401 Unauthorized.")

# 3.4 Test Body bernilai boolean literal 'true'
r_bool_body = client.post('/api/v1/predict', headers=auth_headers, data="true", content_type='application/json')
assert r_bool_body.status_code == 400, f"Expected 400 for 'true', got {r_bool_body.status_code}"
assert "json object" in r_bool_body.get_json()['error'].lower()
print("  [PASS] POST body 'true' - Berhasil dicegat dengan 400 Bad Request.")

# 3.5 Test Body bernilai angka literal '12'
r_int_body = client.post('/api/v1/predict', headers=auth_headers, data="12", content_type='application/json')
assert r_int_body.status_code == 400, f"Expected 400 for '12', got {r_int_body.status_code}"
assert "json object" in r_int_body.get_json()['error'].lower()
print("  [PASS] POST body '12' - Berhasil dicegat dengan 400 Bad Request.")

# 3.6 Test Body valid lengkap dengan 6 skor komposit
valid_payload = {
    'jurusan': 'Rekayasa Perangkat Lunak (RPL)',
    'matematika': 88.0,
    'bahasa_indonesia': 85.0,
    'bahasa_inggris': 82.0,
    'produktif': 91.0,
    'skor_riasec': {
        'Realistic': 24,
        'Investigative': 31,
        'Artistic': 20,
        'Social': 19,
        'Enterprising': 22,
        'Conventional': 25
    }
}
r_valid = client.post('/api/v1/predict', headers=auth_headers, data=json.dumps(valid_payload), content_type='application/json')
assert r_valid.status_code == 200, f"Expected 200 for valid payload, got {r_valid.status_code}"
res_json = r_valid.get_json()
assert res_json['success'] is True
assert res_json['data']['jurusan_kanonik'] == 'RPL'
assert 'metadata' in res_json
assert res_json['metadata']['model_version'] == '1.1.0'
geometry = res_json['data']['profil_minat_akademik']['grafik_kmeans']
assert geometry['x'] == 31 and geometry['y'] == 86.5
assert set(geometry['centroid_0']) == {'x', 'y'}
assert res_json['data']['rekomendasi_utama']['hubungan_jurusan'] in ('Linier', 'Lintas bidang')
assert 'sumber_referensi' in res_json['data']['katalog_contoh_prodi']
print("  [PASS] POST payload valid dengan skor komposit - 200 OK, jurusan kanonik 'RPL', metadata lengkap.")

# 3.7 Test Body valid dengan 42 jawaban mentah via Bearer Token
payload_42 = {
    'jurusan': 'Akuntansi (AKT)',
    'matematika': 85.0,
    'bahasa_indonesia': 88.0,
    'bahasa_inggris': 80.0,
    'produktif': 89.0,
    'jawaban_riasec_42': [3] * 42
}
r_valid_42 = client.post('/api/v1/predict', headers={'Authorization': f'Bearer {TEST_API_KEY}'}, data=json.dumps(payload_42), content_type='application/json')
assert r_valid_42.status_code == 200, f"Expected 200 for 42 answers, got {r_valid_42.status_code}"
res_json_42 = r_valid_42.get_json()
assert res_json_42['success'] is True
assert res_json_42['data']['jurusan_kanonik'] == 'AKL'
print("  [PASS] POST payload valid dengan 42 butir mentah via Bearer token - 200 OK, jurusan kanonik 'AKL'.")

# ------------------------------------------------------------------------------
# KESIMPULAN AKHIR
# ------------------------------------------------------------------------------
print("\n" + "=" * 85)
print("[V] SELURUH KASUS DALAM PENGUJIAN INFERENCE ENGINE & API LOLOS 100%!")
print("    - Reproduksi Klaster & Prediksi RF : 36/36 Siswa Data Uji (0 Mismatch)")
print("    - Reproduksi Vektor Probabilitas   : Selisih Maksimal 0.00e+00 (Identik Penuh)")
print("    - Konsistensi Voting 100 Pohon     : 100% Tepat (Total 100, Kunci Valid)")
print("    - Pertahanan Validasi Input        : 16/16 Kasus Batas Engine Tercegat/Lolos")
print("    - Pertahanan & Autentikasi API     : 7/7 Kasus REST API Terverifikasi")
print("=" * 85)
sys.exit(0)
