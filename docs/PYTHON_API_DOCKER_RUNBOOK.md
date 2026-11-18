# Runbook Deployment Docker — NALARA Career Prediction Python API

Panduan ini mencakup cara membangun, menguji secara lokal, dan men-deploy layanan REST API inferensi karier SMK NALARA (**Two-Stage Hybrid: K-Means 2D `K=2` + Random Forest 100 Trees**, dilatih pada 177 siswa SMKN 1 Bintan Timur untuk 5 jurusan: `RPL`, `AKL`, `PH`, `DKV`, `TB`) berbasis Docker.

---

## 1. Artefak Paket Deployment

Seluruh artefak berada di repositori **Data Mining** (terpisah dari repositori Flutter/Supabase NALARA):

| File | Fungsi |
| --- | --- |
| [Dockerfile](../Dockerfile) | Image produksi `python:3.13-slim`, user non-root `nalara` (`uid=10001`), Gunicorn WSGI (`--preload`), dan `HEALTHCHECK` bawaan |
| [.dockerignore](../.dockerignore) | **Allowlist ketat** (`*` lalu hanya mengizinkan 4 file produksi) agar tidak ada data siswa/PII/Excel/CSV/arsip yang masuk ke build context maupun image |
| [requirements-api.txt](../requirements-api.txt) | Dependensi produksi minimal dengan versi ter-pin (`scikit-learn==1.9.1`, `numpy==2.5.4`, `scipy==1.18.1`, `joblib==1.6.0`, `flask==3.1.3`, `flask-cors==6.0.2`, `gunicorn==23.0.0`) agar cocok 100% dengan bundle joblib |
| [.env.api.example](../.env.api.example) | Template environment variable (`PORT`, `NALARA_API_KEY`, tuning Gunicorn) tanpa kunci asli |
| [pipeline_177_data_mining/api_server.py](../pipeline_177_data_mining/api_server.py) | Endpoint `/health`, `/api/v1/predict`, `/predict` dengan autentikasi `X-API-Key` / `Authorization: Bearer` (*fail-closed* bila `NALARA_API_KEY` kosong) |
| [pipeline_177_data_mining/inference_engine.py](../pipeline_177_data_mining/inference_engine.py) | Engine validasi input, normalisasi `StandardScaler`, K-Means 2D, dan agregasi voting 100 pohon Random Forest |
| [pipeline_177_data_mining/output_model/pipeline_bundle_177.joblib](../pipeline_177_data_mining/output_model/pipeline_bundle_177.joblib) | Bundel model resmi (`634.7 KB`, `scikit-learn 1.9.1`, tanpa objek `pandas`) |

---

## 2. Konfigurasi Environment Variable

Salin `.env.api.example` menjadi `.env.api` untuk uji lokal (`.env.api` sudah masuk `.gitignore` dan **tidak boleh** di-commit):

| Variable | Wajib | Default | Keterangan |
| --- | --- | --- | --- |
| `NALARA_API_KEY` | **Ya** | *(kosong — fail-closed 401)* | Kunci rahasia internal (minimal 32 karakter acak, mis. `openssl rand -hex 32`). Wajib sama persis dengan secret `NALARA_PYTHON_API_KEY` di Supabase Edge Function `predict-career`. |
| `PORT` | Opsional | `5000` | Port HTTP di dalam container. Platform PaaS (Railway/Render/Cloud Run) mengisi `PORT` secara otomatis dan langsung diikuti oleh `CMD` Gunicorn. |
| `WEB_CONCURRENCY` | Opsional | `2` | Jumlah worker proses Gunicorn (`--preload` membagikan memori model antar-worker). Turunkan ke `1` bila memakai kontainer 256 MiB RAM. |
| `GUNICORN_THREADS` | Opsional | `4` | Jumlah thread per worker Gunicorn. |
| `GUNICORN_TIMEOUT` | Opsional | `30` | Batas waktu request worker (detik). |

---

## 3. Build & Pengujian Container Lokal

### 3.1 Build Image
Jalankan dari root repositori Data Mining:
```bash
docker build -t nalara-career-api:1.1.0 .
```
Verifikasi bahwa build context sangat kecil (`~662 KB`) dan audit isi `/app` hanya memuat 4 file produksi tanpa data siswa:
```bash
docker run --rm --entrypoint sh nalara-career-api:1.1.0 -c "id && find /app -maxdepth 3 -type f | sort"
```

### 3.2 Jalankan Container Lokal
```powershell
docker rm -f nalara-career-api 2>$null
docker run -d --name nalara-career-api `
  -e PORT=8088 `
  -e NALARA_API_KEY="ganti-dengan-kunci-uji-lokal" `
  -p 127.0.0.1:18088:8088 `
  nalara-career-api:1.1.0
```
*(Catatan Windows: hindari memetakan port host `5000` jika masuk rentang port yang dicadangkan Hyper-V/Docker; gunakan port host seperti `18088`.)*

### 3.3 Uji Smoke Endpoint
1. **Health Check (`200 OK`)**:
   ```bash
   curl -i http://127.0.0.1:18088/health
   ```
2. **Prediksi Tanpa Kunci (`401 Unauthorized`)**:
   ```bash
   curl -i -X POST http://127.0.0.1:18088/api/v1/predict \
     -H "Content-Type: application/json" \
     -d '{"jurusan":"RPL","matematika":85,"bahasa_indonesia":82,"bahasa_inggris":88,"produktif":90,"jawaban_riasec_42":[5,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4]}'
   ```
3. **Prediksi Valid (`200 OK`)**:
   ```bash
   curl -i -X POST http://127.0.0.1:18088/api/v1/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: ganti-dengan-kunci-uji-lokal" \
     -d '{"jurusan":"RPL","matematika":85,"bahasa_indonesia":82,"bahasa_inggris":88,"produktif":90,"jawaban_riasec_42":[5,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4,4,5,3,3,2,4]}'
   ```
4. **Hentikan & Hapus Container Uji**:
   ```bash
   docker rm -f nalara-career-api
   ```

---

## 4. Kebutuhan Resource Terukur (Hasil Uji Container Lokal)

Pengukuran dilakukan langsung pada image `nalara-career-api:1.0.0` (`WEB_CONCURRENCY=2`, `GUNICORN_THREADS=4`, `PORT=8088`):

| Metrik | Hasil Terukur |
| --- | --- |
| Ukuran build context (`.dockerignore` allowlist) | `661.68 KB` |
| Ukuran image Docker akhir | `546 MB` |
| Waktu *cold-start* hingga `/health` HTTP `200` | `~4.0 detik` (`4018 ms`) |
| Latensi inferensi pertama | `258 ms` |
| Rata-rata latensi inferensi (10 request beruntun) | `126.9 ms/req` |
| Penggunaan RAM (`docker stats` setelah 10 inferensi) | `124.7 MiB` (`19 PIDs`) |
| Penggunaan CPU idle | `0.05%` |
| Peringatan versi `scikit-learn` / Traceback | `0` |

### Rekomendasi Spesifikasi Hosting
- **Staging / Produksi Standar (Disarankan)**: **512 MiB RAM**, **0.5 vCPU**, `WEB_CONCURRENCY=2`, `GUNICORN_THREADS=4` (memberi ruang aman ~4× dari pemakaian terukur `124.7 MiB`).
- **Tier Hemat / Minimal**: **256 MiB RAM**, **0.25–0.5 vCPU** dengan menurunkan `WEB_CONCURRENCY=1`, `GUNICORN_THREADS=4`.
- **Persyaratan Jaringan**: wajib HTTPS publik (disediakan otomatis oleh PaaS seperti Railway, Render, Fly.io, atau Cloud Run; untuk VPS gunakan reverse proxy TLS seperti Caddy/Nginx) karena Edge Function Supabase memanggil endpoint ini dari cloud.

---

## 5. Integrasi ke Supabase Staging / Produksi

Setelah container ter-deploy pada URL HTTPS (mis. `https://<host-api>/api/v1/predict`):

1. Pastikan `/health` mengembalikan `200 OK` dan `/api/v1/predict` tanpa kunci mengembalikan `401`.
2. Pasang secret pada project Supabase target (di repositori NALARA):
   ```bash
   supabase secrets set --project-ref <PROJECT-REF> \
     NALARA_PYTHON_API_URL="https://<host-api>/api/v1/predict" \
     NALARA_PYTHON_API_KEY="<nilai-sama-dengan-NALARA_API_KEY>"
   ```
3. Terapkan migrasi `supabase/career_prediction_hybrid.sql` lalu deploy Edge Function `predict-career`:
   ```bash
   supabase functions deploy predict-career --project-ref <PROJECT-REF> --no-verify-jwt
   ```
4. Jangan pernah menyimpan `NALARA_API_KEY` di aplikasi Flutter, `.env.api.example`, atau repositori Git.


## Revisi model 1.1.0

Bundle memakai scaler dan centroid dari 141 data latih. API mengirim geometri dua klaster dan hubungan rumpun?jurusan. Metrik resource di atas berasal dari pengukuran image 1.0.0. Uji image baru secara lokal dengan `python tools/test_api_docker.py`; belum ada deployment remote dari revisi ini.
