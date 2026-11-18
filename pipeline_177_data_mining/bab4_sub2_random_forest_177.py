"""IV.2: RF 100 pohon dan ekspor bundle terintegrasi.
Rumus dan sumber data bersama: research_pipeline.py.
"""
from threadpoolctl import threadpool_limits
from research_pipeline import fit_forest

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        fit_forest()
