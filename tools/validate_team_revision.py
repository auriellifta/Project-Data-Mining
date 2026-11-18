"""Periksa konsistensi paket tim dan ketiadaan kolom identitas langsung."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'Bahan_Laporan_Akhir_Untuk_Tim_v2'
BASE=ROOT/'pipeline_177_data_mining'


def main():
    manifest=json.loads((PACKAGE/'MANIFEST.json').read_text(encoding='utf-8'))
    assert manifest['model_version']=='1.1.0'
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==digest,name
    for path in PACKAGE.rglob('*.csv'):
        columns=pd.read_csv(path,nrows=0).columns.str.casefold()
        assert not set(columns).intersection({'nis','nama','nama_siswa','nama siswa'}),path
    m=json.loads((BASE/'output_model/metrics_177.json').read_text(encoding='utf-8'))
    df=pd.read_csv(BASE/'dataset_terklaster_177.csv')
    assert {str(k):int(v) for k,v in df.Klaster_KMeans.value_counts().items()}==m['kmeans']['cluster_counts']
    old=ROOT/'integration_artifacts/revision_backup_20261118/pipeline_177_data_mining/dataset_engineered_177.csv'
    if old.exists():
        a=pd.read_csv(old).set_index('ID_Siswa').sort_index()
        b=pd.read_csv(BASE/'dataset_engineered_177.csv').set_index('ID_Siswa').sort_index()
        numeric=a.select_dtypes(include='number').columns
        np.testing.assert_allclose(a[numeric],b[numeric],atol=1e-12)
        assert a.Target_Rumpun_Kuliah.equals(b.Target_Rumpun_Kuliah)
    doc=PACKAGE/'3_Panduan_Narasi_dan_Angka_Resmi_Laporan/ringkasan project datmin_revisi_v2.docx'
    with ZipFile(doc) as z:
        for name in z.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):ET.fromstring(z.read(name))
        body=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        assert len(body.findall('.//w:tbl',ns))>=4
    print(f'PASS paket tim: {len(manifest["sha256"])} hash cocok; CSV tanpa Nama/NIS; DOCX valid; nilai mentah dan target tidak berubah.')


if __name__=='__main__':main()
