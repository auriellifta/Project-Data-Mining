"""III.6: split 141/36 sebelum fit scaler.
Rumus dan sumber data bersama: research_pipeline.py.
"""
from threadpoolctl import threadpool_limits
from research_pipeline import transform_dataset

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        transform_dataset()
