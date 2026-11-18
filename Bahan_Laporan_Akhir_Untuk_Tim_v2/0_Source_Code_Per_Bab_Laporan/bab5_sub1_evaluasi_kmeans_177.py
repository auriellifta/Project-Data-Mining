"""V.1: evaluasi K-Means pada bundle revisi.
Rumus dan sumber data bersama: research_pipeline.py.
"""
from threadpoolctl import threadpool_limits
from research_pipeline import plot_kmeans

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        plot_kmeans()
