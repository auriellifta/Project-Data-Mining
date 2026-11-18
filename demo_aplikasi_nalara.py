"""Demo CLI memakai engine, scaler, dan centroid yang sama dengan REST API."""
from pipeline_177_data_mining.inference_engine import NalaraInferenceEngine

def inferensi_siswa(nama_siswa,jurusan_smk,nilai_rapor,skor_riasec):
    engine=NalaraInferenceEngine()
    scores=dict(zip(engine.RIASEC_DIMENSIONS,skor_riasec))
    return engine.predict_student(jurusan_smk,*nilai_rapor,scores)

if __name__=='__main__':
    import json
    print(json.dumps(inferensi_siswa('Simulasi','RPL',[85,82,88,90],[24,31,20,23,22,25]),ensure_ascii=False,indent=2))
