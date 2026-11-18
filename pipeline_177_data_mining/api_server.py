# ==============================================================================
# PROYEK DATA MINING NALARA - REST API SERVICE
# File: pipeline_177_data_mining/api_server.py
# Layanan REST API Prediksi Rumpun Studi & Diagnosa Klaster NALARA
# ==============================================================================

import os
import sys
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from inference_engine import NalaraInferenceEngine
except ImportError:
    from pipeline_177_data_mining.inference_engine import NalaraInferenceEngine

app = Flask(__name__)
CORS(app)  # Mengizinkan akses cross-origin

# Kunci rahasia API internal untuk komunikasi aman antar-backend (Supabase -> Python API)
# Wajib disetel melalui environment variable NALARA_API_KEY (tanpa fallback hardcoded di source code)
INTERNAL_API_KEY = os.environ.get('NALARA_API_KEY')
if not INTERNAL_API_KEY or not INTERNAL_API_KEY.strip():
    INTERNAL_API_KEY = None
    print("[PERINGATAN KEAMANAN] Environment variable 'NALARA_API_KEY' belum disetel. Seluruh akses prediksi akan ditolak dengan 401 Unauthorized sampai kunci disetel.")
else:
    print("[V] Autentikasi internal API terkonfigurasi via environment variable NALARA_API_KEY.")

# Inisialisasi model engine secara singleton saat server startup
try:
    engine = NalaraInferenceEngine()
    print("[V] NalaraInferenceEngine siap melayani prediksi!")
except Exception as e:
    print(f"[ERROR] Gagal memuat inference engine: {e}")
    sys.exit(1)

# Katalog Referensi Kurasi Contoh Program Studi per Rumpun (Non-ranking / Kurasi Editorial)
KATALOG_CONTOH_PRODI = {
    'Sains & Teknologi Informasi': [
        'S1 Informatika / Ilmu Komputer',
        'S1 Sistem Informasi',
        'D4 Rekayasa Perangkat Lunak Aplikasi',
        'S1 Sains Data',
        'D4 Teknologi Rekayasa Komputer'
    ],
    'Ekonomi, Bisnis & Akuntansi': [
        'S1 Akuntansi',
        'S1 Manajemen Bisnis',
        'D4 Akuntansi Perpajakan',
        'S1 Bisnis Digital',
        'D4 Keuangan Perbankan'
    ],
    'Seni, Desain & Industri Kreatif': [
        'S1 Desain Komunikasi Visual (DKV)',
        'D4 Desain Mode & Busana',
        'S1 Animasi & Desain Game',
        'S1 Seni Media Rekam',
        'D4 Kriya Seni & Desain Grafis'
    ],
    'Pariwisata & Hospitality': [
        'D4 Pengelolaan Perhotelan',
        'S1 Manajemen Pariwisata',
        'D3 Seni Kuliner & Patiseri',
        'D4 Manajemen Bisnis Konvensi & Event (MICE)',
        'S1 Destinasi Pariwisata'
    ],
    'Sosial, Humaniora & Pendidikan': [
        'S1 Bimbingan dan Konseling (BK)',
        'S1 Ilmu Komunikasi',
        'S1 Pendidikan Bahasa Inggris',
        'S1 Psikologi',
        'S1 Administrasi Publik'
    ]
}

def verify_api_key(req):
    """
    Memeriksa kredensial autentikasi API Key pada header.
    Menerima header 'X-API-Key' atau 'Authorization: Bearer <key>'.
    Jika server tidak memiliki NALARA_API_KEY yang terkonfigurasi, tolak seluruh akses.
    """
    if not INTERNAL_API_KEY:
        return False
        
    api_key = req.headers.get('X-API-Key')
    if not api_key:
        auth_header = req.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            api_key = auth_header[7:].strip()
            
    return bool(api_key and api_key == INTERNAL_API_KEY)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint untuk pemeriksaan status keaktifan layanan (Health Check publik)."""
    return jsonify({
        'status': 'healthy',
        'service': 'NALARA Career Prediction API',
        'version': engine.MODEL_METADATA['model_version'],
        'pipeline': engine.MODEL_METADATA['pipeline_version'],
        'target_classes': engine.target_classes,
        'canonical_jurusan': list(engine.CANONICAL_JURUSAN),
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
    }), 200

@app.route('/predict', methods=['POST'])
@app.route('/api/v1/predict', methods=['POST'])
def predict_career():
    """
    Endpoint utama prediksi rumpun studi dan diagnosa profil minat-akademik siswa.
    Memerlukan autentikasi internal header 'X-API-Key' atau 'Authorization: Bearer <key>'.
    Mendukung input 'skor_riasec' (6 dimensi agregat) ATAU 'jawaban_riasec_42' / 'raw_42_answers' (42 butir raw Likert).
    Mendukung field nilai kejuruan sebagai 'produktif' ATAU 'kejuruan'.
    """
    # 1. Autentikasi Backend Tepercaya
    if not verify_api_key(request):
        return jsonify({
            'success': False,
            'error': 'Akses ditolak. Layanan inferensi ini memerlukan API Key autentikasi yang sah pada header X-API-Key atau Bearer Token.'
        }), 401

    # 2. Validasi Body JSON
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            'success': False,
            'error': f'Payload request wajib berupa JSON object (dictionary). Diterima tipe: {type(data).__name__}.'
        }), 400

    required_fields = ['jurusan', 'matematika', 'bahasa_indonesia', 'bahasa_inggris']
    missing_fields = [f for f in required_fields if f not in data]
    
    # Validasi kejuruan / produktif dengan dukungan alias
    produktif_val = data.get('produktif')
    if produktif_val is None:
        produktif_val = data.get('kejuruan')
    if produktif_val is None:
        missing_fields.append('produktif (atau kejuruan)')

    if missing_fields:
        return jsonify({
            'success': False,
            'error': f'Field berikut wajib disertakan: {", ".join(missing_fields)}'
        }), 400

    # Cek ketersediaan input RIASEC (skor agregat atau 42 jawaban mentah dengan alias)
    riasec_input = data.get('skor_riasec')
    if riasec_input is None:
        riasec_input = data.get('jawaban_riasec_42')
    if riasec_input is None:
        riasec_input = data.get('raw_42_answers')
        
    if riasec_input is None:
        return jsonify({
            'success': False,
            'error': "Wajib menyertakan salah satu input RIASEC: 'skor_riasec' (6 skor dimensi kumulatif) atau 'jawaban_riasec_42' / 'raw_42_answers' (daftar 42 jawaban skala 1-5)."
        }), 400

    try:
        hasil = engine.predict_student(
            jurusan=data['jurusan'],
            matematika=data['matematika'],
            bahasa_indonesia=data['bahasa_indonesia'],
            bahasa_inggris=data['bahasa_inggris'],
            produktif=produktif_val,
            riasec_scores=riasec_input
        )

        rumpun = hasil['rumpun_terprediksi']
        contoh_prodi = KATALOG_CONTOH_PRODI.get(rumpun, [])

        response_payload = {
            'success': True,
            'data': {
                'jurusan_kanonik': hasil['jurusan_kanonik'],
                'rekomendasi_utama': {
                    'rumpun_studi': rumpun,
                    'tingkat_keyakinan_pct': hasil['confidence_score_pct'],
                    'sebaran_probabilitas': hasil['probabilitas_per_rumpun'],
                    'voting_pohon_100': hasil['voting_pohon']
                },
                'profil_minat_akademik': {
                    **hasil['klaster_kmeans'],
                    'jarak_euclidean': hasil['jarak_euclidean']
                },
                'radar_riasec_komposit': hasil['skor_riasec_komposit'],
                'katalog_contoh_prodi': {
                    'catatan': 'Contoh program studi referensi dalam rumpun terpilih (non-ranking, katalog kurasi editorial)',
                    'sumber_referensi': (
                        'Katalog kurasi internal proyek NALARA berdasarkan profil nomenklatur program studi perguruan tinggi di Indonesia (non-ranking, disediakan sebagai bahan referensi eksplorasi).'
                    ),
                    'daftar_prodi': contoh_prodi
                }
            },
            'metadata': {
                'model_version': hasil['model_metadata']['model_version'],
                'pipeline_version': hasil['model_metadata']['pipeline_version'],
                'algorithm': hasil['model_metadata']['algorithm'],
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        }
        return jsonify(response_payload), 200

    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 422
    except Exception as ex:
        return jsonify({
            'success': False,
            'error': f'Terjadi kesalahan internal pada pemrosesan inferensi: {str(ex)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Menjalankan NALARA API Service pada http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
