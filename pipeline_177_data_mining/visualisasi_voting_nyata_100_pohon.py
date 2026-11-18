"""V.3: seluruh prediksi tercakup dalam histogram 0?100.
Rumus dan sumber data bersama: research_pipeline.py.
"""
from threadpoolctl import threadpool_limits
from research_pipeline import plot_votes

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        plot_votes()
