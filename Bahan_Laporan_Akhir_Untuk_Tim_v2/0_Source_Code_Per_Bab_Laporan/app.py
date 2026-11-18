"""Dashboard demonstrasi penelitian; memakai bundle inferensi yang sama dengan API."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pipeline_177_data_mining.inference_engine import NalaraInferenceEngine

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'pipeline_177_data_mining'
OUT = BASE / 'output_model'
st.set_page_config(page_title='NALARA ? Studi Lanjut SMK', page_icon='??', layout='wide')

@st.cache_resource
def resources(bundle_mtime):
    return NalaraInferenceEngine()

engine = resources((OUT / 'pipeline_bundle_177.joblib').stat().st_mtime_ns)
st.title('NALARA ? Eksplorasi Studi Lanjut SMK')
st.caption(f"Model {engine.MODEL_METADATA['model_version']} ? K-Means K=2 ? RF 100 pohon")
sim, evaluation, data_tab, methodology = st.tabs(['Simulasi & hasil', 'Evaluasi model', 'Data uji', 'Metodologi'])
with sim:
    with st.form('student'):
        major = st.selectbox('Jurusan SMK', ['RPL','AKL','PH','DKV','TB'])
        cols = st.columns(4)
        grades = [col.number_input(label, min_value=0., max_value=100., value=85.)
                  for col, label in zip(cols,['Matematika','B. Indonesia','B. Inggris','Produktif'])]
        columns = st.columns(3)
        scores = {dim:columns[j % 3].number_input(dim, min_value=7,max_value=35,value=21)
                  for j,dim in enumerate(engine.RIASEC_DIMENSIONS)}
        submit = st.form_submit_button('Jalankan K-Means dan Random Forest')
    if submit:
        result = engine.predict_student(major,*grades,scores)
        st.subheader('Tahap 1 ? K-Means 2D')
        profile = result['klaster_kmeans']; g = profile['grafik_kmeans']
        st.write(f"Klaster {profile['kode']} ? **{profile['label']}**")
        a,b = st.columns(2)
        a.metric(f"X ? {g['dimensi_minat']}",f"{g['x']:.0f}/35")
        b.metric('Y ? Rata-rata empat pelajaran',f"{g['y']:.2f}")
        fig,ax=plt.subplots(figsize=(7,4))
        for code,color in [(0,'#cf791f'),(1,'#21956c')]:
            center=g[f'centroid_{code}']; ax.scatter(center['x'],center['y'],marker='X',s=150,color=color,label=f'Centroid {code}')
        ax.scatter(g['x'],g['y'],marker='*',s=180,color='#2355c0',label='Siswa')
        ax.set(xlim=(7,35),ylim=(0,100),xlabel='X: minat kejuruan (7?35)',ylabel='Y: rata-rata 4 pelajaran (0?100)')
        ax.legend();st.pyplot(fig);plt.close(fig)
        st.caption(f"Jarak Z-score: C0={g['jarak_centroid_0']:.4f}; C1={g['jarak_centroid_1']:.4f}")
        st.info(profile['catatan_konseling'])
        st.subheader('Tahap 2 ? Random Forest')
        st.write(f"**{result['rumpun_terprediksi']}** ? {result['hubungan_jurusan']}")
        st.metric('Tingkat keyakinan model',f"{result['confidence_score_pct']:.1f}%")
        table=pd.DataFrame({'Rumpun':result['probabilitas_per_rumpun'].keys(),
            'Probabilitas':result['probabilitas_per_rumpun'].values(),
            'Suara dari 100 pohon':result['voting_pohon'].values()}).set_index('Rumpun')
        st.bar_chart(table['Probabilitas']);st.dataframe(table)
with evaluation:
    import json
    metrics=json.loads((OUT/'metrics_177.json').read_text(encoding='utf-8'))
    st.metric('Akurasi holdout',f"{metrics['test_accuracy']:.2%}")
    st.write(f"{metrics['test_correct']}/{metrics['test_rows']} benar terhadap target proxy berbasis aturan.")
    st.write('Kelas tanpa contoh di data uji:',metrics['untested_classes'])
    for name in ['scatter_kmeans_2d_177.png','evaluasi_silhouette_kmeans_177.png','confusion_matrix_rf_177.png','feature_importance_rf_177.png','hasil_voting_nyata_100_pohon_177.png']:
        st.image(str(OUT/name))
with data_tab:
    df=pd.read_csv(BASE/'hasil_prediksi_rf_177.csv')
    st.dataframe(df)
    st.download_button('Unduh hasil uji anonim',df.to_csv(index=False).encode('utf-8'),'hasil_uji_177.csv','text/csv')
with methodology:
    st.markdown((ROOT/'README.md').read_text(encoding='utf-8'))
