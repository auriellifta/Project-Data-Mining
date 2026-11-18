"""V.2: evaluasi lengkap, kelas tanpa support dinyatakan belum diuji.
Rumus dan sumber data bersama: research_pipeline.py.
"""
from threadpoolctl import threadpool_limits
from research_pipeline import evaluate, plot_forest

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        evaluate()
        plot_forest()
