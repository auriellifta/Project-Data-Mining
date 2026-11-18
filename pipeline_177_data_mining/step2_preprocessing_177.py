"""Menjalankan enam langkah preprocessing dalam urutan sumber yang sama."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    for number in range(1, 7):
        paths = sorted(base.glob(f'bab3_sub{number}_*.py'))
        assert len(paths) == 1
        runpy.run_path(str(paths[0]), run_name='__main__')
