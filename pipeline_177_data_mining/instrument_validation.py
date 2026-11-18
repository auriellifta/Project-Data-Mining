"""Diagnostik instrumen: alpha, item-total, item-rest, dan korelasi eksploratif.

Korelasi item-total memasukkan butir itu sendiri. Item-rest mengeluarkannya.
Keduanya tidak menggantikan validasi isi/konstruk oleh pakar.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUTPUT = BASE / 'output_eda'
DIMENSIONS = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
LIKERT = {'Sangat Tidak Sesuai': 1, 'Tidak Sesuai': 2, 'Cukup Sesuai': 3, 'Sesuai': 4, 'Sangat Sesuai': 5}


def validate_instrument():
    raw = pd.read_excel(ROOT / 'Kuesioner Minat, Kemampuan, dan Arah Karier Siswa SMKN V2 (Jawaban).xlsx')
    n = len(raw); critical_t = t.ppf(.975, n-2)
    critical_r = float(np.sqrt(critical_t**2 / (critical_t**2+n-2)))
    items_result, alphas = [], []
    for offset, dimension in enumerate(DIMENSIONS):
        positions = [5+offset,11+offset,17+offset,23+offset,29+offset,36+offset,43+offset]
        items = raw.iloc[:, positions].apply(lambda s: s.map(LIKERT))
        assert not items.isna().any().any()
        total = items.sum(axis=1); total_variance = float(total.var(ddof=1))
        variance_sum = float(items.var(ddof=1).sum())
        alpha = 7/6*(1-variance_sum/total_variance)
        alphas.append({'Dimensi RIASEC': dimension, 'Jumlah Butir (k)': 7,
                       'Jumlah Varians Butir (Sum Var)': variance_sum,
                       'Varians Total (Var Total)': total_variance, 'Cronbach Alpha': alpha})
        for j, column in enumerate(items.columns):
            score = items[column]
            whole_r, whole_p = pearsonr(score, total)
            rest_r, rest_p = pearsonr(score, total-score)
            items_result.append({'Dimensi': dimension, 'Kode Butir': f'B{j+1}',
                'Pernyataan Instrumen': str(column), 'r_item_total': float(whole_r),
                'p_item_total': float(whole_p), 'r_item_rest_corrected': float(rest_r),
                'p_item_rest_corrected': float(rest_p), 'r_acuan_dua_arah_005': critical_r,
                'Catatan': 'Korelasi item-rest terdeteksi' if rest_r>=critical_r and rest_p<.05 else 'Perlu telaah butir'})
    items_table, alpha_table = pd.DataFrame(items_result), pd.DataFrame(alphas)
    integrated = pd.read_csv(BASE / 'dataset_terintegrasi_177.csv')
    correlations = []
    for grade in ['Matematika','Bahasa_Indonesia','Bahasa_Inggris','Produktif']:
        for dimension in DIMENSIONS:
            r, p = pearsonr(integrated[grade], integrated[dimension])
            correlations.append({'Mapel':grade,'Dimensi':dimension,'r':float(r),'p_value':float(p)})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    items_table.to_csv(OUTPUT/'diagnostik_item_177.csv',index=False)
    alpha_table.to_csv(OUTPUT/'reliabilitas_177.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUTPUT/'tabel_korelasi_rapor_riasec.csv',index=False)
    notes = pd.DataFrame({'Keterangan':[
        'Item-total memasukkan skor butir yang sedang diuji; utamakan item-rest untuk diagnostik konsistensi.',
        'p-value korelasi adalah eksploratif, tanpa koreksi pengujian berganda; bukan bukti kausalitas.',
        'Alpha rendah harus diungkapkan; standardisasi Z-score tidak memperbaiki reliabilitas instrumen.',
        'Diagnostik ini tidak menetapkan semua butir valid dan tidak memvalidasi target rekomendasi RF.',
        'Instrumen tetap 42 butir untuk kompatibilitas versi; perubahan instrumen memerlukan kajian terpisah.'
    ]})
    with pd.ExcelWriter(OUTPUT/'Hasil_Uji_Validitas_dan_Reliabilitas_Instrumen_RIASEC.xlsx') as writer:
        items_table.to_excel(writer,sheet_name='Diagnostik_42_Butir',index=False)
        alpha_table.to_excel(writer,sheet_name='Reliabilitas_6_Dimensi',index=False)
        pd.DataFrame(correlations).to_excel(writer,sheet_name='Korelasi_Eksploratif',index=False)
        notes.to_excel(writer,sheet_name='Batas_Interpretasi',index=False)
    summary = {'n':n,'r_critical':critical_r,
               'item_rest_flagged':int((items_table.Catatan=='Perlu telaah butir').sum()),
               'alpha':dict(zip(alpha_table['Dimensi RIASEC'],alpha_table['Cronbach Alpha']))}
    (OUTPUT/'instrument_metrics_177.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(alpha_table[['Dimensi RIASEC','Cronbach Alpha']].to_string(index=False))
    print('Butir yang perlu telaah berdasar item-rest:',summary['item_rest_flagged'])
    return summary


if __name__=='__main__':
    validate_instrument()
