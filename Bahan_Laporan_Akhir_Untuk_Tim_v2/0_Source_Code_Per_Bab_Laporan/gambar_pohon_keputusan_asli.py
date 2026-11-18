"""V.3: pohon dibaca langsung dari estimator tersimpan.
Rumus dan sumber data bersama: research_pipeline.py.
"""
from threadpoolctl import threadpool_limits
from research_pipeline import plot_actual_tree

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        plot_actual_tree()
