"""Build dan smoke test image API secara lokal; container selalu dibersihkan."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
IMAGE='nalara-career-api:1.1.0'


def docker(*args, capture=False, env=None):
    return subprocess.run(['docker',*args],cwd=ROOT,env=env,check=True,
                          text=True,capture_output=capture).stdout


def main():
    docker('build','-t',IMAGE,'.')
    key=secrets.token_urlsafe(32)
    env={**os.environ,'NALARA_API_KEY':key}
    container=docker('run','-d','--name',f'nalara-revision-check-{os.getpid()}',
                     '-p','127.0.0.1::5000','-e','NALARA_API_KEY',IMAGE,capture=True,env=env).strip()
    try:
        port=docker('port',container,'5000',capture=True).strip().rsplit(':',1)[1]
        base=f'http://127.0.0.1:{port}'
        deadline=time.monotonic()+45
        while True:
            try:
                with urlopen(base+'/health',timeout=2) as response:health=json.load(response)
                break
            except (URLError,TimeoutError,ConnectionError):
                if time.monotonic()>deadline:raise RuntimeError('API container tidak siap dalam 45 detik')
                time.sleep(.25)
        assert health['version']=='1.1.0'
        payload={'jurusan':'RPL','matematika':0,'bahasa_indonesia':80,'bahasa_inggris':80,'produktif':80,
                 'skor_riasec':dict(zip(['Realistic','Investigative','Artistic','Social','Enterprising','Conventional'],[24,31,20,23,22,25]))}
        def predict(data,authorized=True):
            headers={'Content-Type':'application/json'}
            if authorized:headers['X-API-Key']=key
            try:
                with urlopen(Request(base+'/api/v1/predict',data=json.dumps(data).encode(),headers=headers),timeout=10) as response:
                    return response.status,json.load(response)
            except HTTPError as error:return error.code,json.load(error)
        assert predict(payload,False)[0]==401
        assert predict({**payload,'matematika':150})[0]==422
        status,response=predict(payload);assert status==200
        g=response['data']['profil_minat_akademik']['grafik_kmeans']
        assert g['x']==31 and g['y']==60
        assert sum(response['data']['rekomendasi_utama']['voting_pohon_100'].values())==100
        assert response['metadata']['model_version']=='1.1.0'
        print('PASS Docker API 1.1.0: health, auth 401, invalid 422, inferensi 200, geometri X/Y dan 100 suara.')
    finally:
        docker('rm','-f',container,capture=True)
        print('Container smoke test dibersihkan.')


if __name__=='__main__':main()
