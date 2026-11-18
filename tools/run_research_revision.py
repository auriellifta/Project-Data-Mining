"""Reproduksi hasil penelitian dan paket tim revisi, dari root repo."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def run(relative):
    subprocess.run([sys.executable,str(ROOT/relative)],cwd=ROOT,check=True)

if __name__=='__main__':
    run('pipeline_177_data_mining/step2_preprocessing_177.py')
    run('pipeline_177_data_mining/research_pipeline.py')
    run('pipeline_177_data_mining/instrument_validation.py')
    run('tools/export_team_revision.py')
