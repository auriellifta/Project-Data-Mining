"""IV.1: K-Means K=2, fit hanya bagian latih.
Rumus dan sumber data bersama: research_pipeline.py.
"""
from threadpoolctl import threadpool_limits
from research_pipeline import fit_kmeans

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        fit_kmeans()
