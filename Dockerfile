# ==============================================================================
# NALARA CAREER PREDICTION PYTHON API - PRODUCTION DOCKERFILE
# Two-Stage Hybrid Model: K-Means 2D (K=2) + Random Forest (100 Trees)
# ==============================================================================
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5000 \
    WEB_CONCURRENCY=2 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=30

WORKDIR /app

# Pasang dependensi produksi minimal (tanpa paket EDA/Streamlit/Matplotlib/Pandas)
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Salin HANYA artefak inferensi & model resmi (terjamin oleh allowlist .dockerignore)
COPY pipeline_177_data_mining/inference_engine.py ./pipeline_177_data_mining/inference_engine.py
COPY pipeline_177_data_mining/api_server.py ./pipeline_177_data_mining/api_server.py
COPY pipeline_177_data_mining/output_model/pipeline_bundle_177.joblib ./pipeline_177_data_mining/output_model/pipeline_bundle_177.joblib

# Jalankan sebagai pengguna non-root demi keamanan produksi
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin nalara \
    && chown -R nalara:nalara /app
USER nalara

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, sys, urllib.request; port = os.environ.get('PORT', '5000'); resp = urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4); sys.exit(0 if resp.status == 200 else 1)"

# Gunicorn dengan --preload memuat NalaraInferenceEngine sekali sebelum fork worker (hemat RAM)
CMD ["sh", "-c", "exec gunicorn --chdir /app/pipeline_177_data_mining --bind 0.0.0.0:${PORT:-5000} --workers ${WEB_CONCURRENCY:-2} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-30} --preload --access-logfile - --error-logfile - api_server:app"]
