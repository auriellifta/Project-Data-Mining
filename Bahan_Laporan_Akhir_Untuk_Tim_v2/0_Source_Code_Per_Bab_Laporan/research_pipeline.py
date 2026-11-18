"""Sumber tunggal pelatihan, evaluasi, dan gambar penelitian NALARA.

Rancangan tetap: X=minat kejuruan RIASEC, Y=rerata empat mapel, K=2,
RF=100 pohon. Semua estimator dipelajari hanya dari bagian latih.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             davies_bouldin_score, silhouette_samples, silhouette_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'output_model'
VERSION = '1.1.0'
PIPELINE_VERSION = 'pipeline_177_smk_train_only_v1_1'
DIMENSIONS = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
GRADES = ['Matematika', 'Bahasa_Indonesia', 'Bahasa_Inggris', 'Produktif']
NUMERIC = GRADES + ['Rata_Rapor'] + DIMENSIONS + ['Skor_Minat_Kejuruan']
KM_FEATURES = ['Skor_Minat_Kejuruan_scaled', 'Rata_Rapor_scaled']
RF_FEATURES = [c + '_scaled' for c in GRADES + DIMENSIONS] + ['Klaster_KMeans']
TARGET = 'Target_Rumpun_Kuliah'
CLUSTER_NAMES = {0: 'Tidak Sesuai', 1: 'Sesuai'}
SHORT_CLASSES = {'Ekonomi, Bisnis & Akuntansi': 'Bisnis', 'Pariwisata & Hospitality': 'Pariwisata',
                 'Sains & Teknologi Informasi': 'Sains & IT', 'Seni, Desain & Industri Kreatif': 'Seni & Desain',
                 'Sosial, Humaniora & Pendidikan': 'Sosial'}
LABEL_NOTE = ('Sesuai/Tidak Sesuai adalah label operasional dua kelompok berdasarkan '
              'minat kejuruan dan rerata rapor; gunakan bersama Guru BK. '
              'Label tidak mensyaratkan dua ambang minimum dan tidak memvalidasi kecocokan karier.')


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_preprocessing():
    return joblib.load(OUTPUT / 'preprocessing_177.joblib')


def split_indices(df):
    return train_test_split(np.arange(len(df)), test_size=.2, stratify=df[TARGET], random_state=42)


def scaled_frame(scaler, frame):
    return pd.DataFrame(scaler.transform(frame[NUMERIC]), index=frame.index,
                        columns=[c + '_scaled' for c in NUMERIC])


def transform_dataset():
    df = pd.read_csv(BASE / 'dataset_engineered_177.csv')
    assert len(df) == 177 and df.ID_Siswa.is_unique
    assert not df[NUMERIC].isna().any().any() and np.isfinite(df[NUMERIC]).all().all()
    train, test = split_indices(df)
    scaler = StandardScaler().fit(df.iloc[train][NUMERIC])
    assert set(train).isdisjoint(test)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = {'scaler': scaler, 'train_ids': df.iloc[train].ID_Siswa.tolist(),
                'test_ids': df.iloc[test].ID_Siswa.tolist(), 'numeric': NUMERIC,
                'model_version': VERSION}
    joblib.dump(artifact, OUTPUT / 'preprocessing_177.joblib')
    pd.concat([df[['ID_Siswa', 'Jurusan', TARGET] + NUMERIC], scaled_frame(scaler, df)], axis=1).to_csv(
        BASE / 'dataset_siap_model_177.csv', index=False)
    write_json(OUTPUT / 'split_177.json', {k: artifact[k] for k in ['train_ids', 'test_ids', 'model_version']})
    print(f'Scaler fit pada {len(train)} data latih; transform {len(test)} data uji. Versi {VERSION}.')


def canonical_clusters(kmeans, values, invert):
    raw = kmeans.predict(values)
    return 1 - raw if invert else raw


def fit_kmeans():
    df = pd.read_csv(BASE / 'dataset_siap_model_177.csv')
    prep = load_preprocessing()
    mask = df.ID_Siswa.isin(prep['train_ids'])
    # Urutan sampel sama dengan split latih, bukan urutan tabel seluruh siswa.
    training = df.set_index('ID_Siswa').loc[prep['train_ids'], KM_FEATURES]
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(training)
        invert = bool(km.cluster_centers_[0].sum() > km.cluster_centers_[1].sum())
        df['Klaster_KMeans'] = canonical_clusters(km, df[KM_FEATURES], invert)
    df['Status_Kesesuaian'] = df.Klaster_KMeans.map(CLUSTER_NAMES)
    df.to_csv(BASE / 'dataset_terklaster_177.csv', index=False)
    joblib.dump({'kmeans': km, 'invert_clusters': invert}, OUTPUT / 'kmeans_177.joblib')
    print('K-Means fit hanya pada data latih; X=minat kejuruan, Y=rerata empat mapel, K=2.')
    print(df.Status_Kesesuaian.value_counts().to_string())
    print(LABEL_NOTE)
    return df


def geometry(scaler, km, invert):
    centers = km.cluster_centers_[::-1] if invert else km.cluster_centers_
    pos = [NUMERIC.index('Skor_Minat_Kejuruan'), NUMERIC.index('Rata_Rapor')]
    raw = centers * scaler.scale_[pos] + scaler.mean_[pos]
    return [{'kode': code, 'label': CLUSTER_NAMES[code],
             'x': float(raw[code, 0]), 'y': float(raw[code, 1]),
             'x_scaled': float(centers[code, 0]), 'y_scaled': float(centers[code, 1])}
            for code in [0, 1]]


def fit_hybrid_in_memory(df, train, test, with_cluster=True):
    """Juga dipakai CV: fit ulang seluruh transformasi di bagian latih fold."""
    train_df, test_df = df.iloc[train], df.iloc[test]
    scaler = StandardScaler().fit(train_df[NUMERIC])
    tr, te = scaled_frame(scaler, train_df), scaled_frame(scaler, test_df)
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(tr[KM_FEATURES])
    invert = bool(km.cluster_centers_[0].sum() > km.cluster_centers_[1].sum())
    for frame in [tr, te]:
        frame['Klaster_KMeans'] = canonical_clusters(km, frame[KM_FEATURES], invert)
    features = RF_FEATURES if with_cluster else RF_FEATURES[:-1]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(tr[features], train_df[TARGET])
    return {'scaler': scaler, 'kmeans': km, 'invert_clusters': invert, 'rf': rf,
            'prediction': rf.predict(te[features]), 'test_features': te[features]}


def fit_forest():
    df = pd.read_csv(BASE / 'dataset_terklaster_177.csv')
    prep, stage1 = load_preprocessing(), joblib.load(OUTPUT / 'kmeans_177.joblib')
    indexed = df.set_index('ID_Siswa')
    train, test = indexed.loc[prep['train_ids']], indexed.loc[prep['test_ids']]
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42, n_jobs=1)
    rf.fit(train[RF_FEATURES], train[TARGET])
    heldout = test[['Jurusan', 'Status_Kesesuaian', TARGET]].copy().reset_index()
    heldout['Prediksi_Random_Forest'] = rf.predict(test[RF_FEATURES])
    heldout['Confidence_Score (%)'] = rf.predict_proba(test[RF_FEATURES]).max(axis=1) * 100
    heldout['Status_Prediksi'] = np.where(heldout[TARGET] == heldout.Prediksi_Random_Forest, 'BENAR', 'SALAH')
    heldout.to_csv(BASE / 'hasil_prediksi_rf_177.csv', index=False)
    joblib.dump(rf, OUTPUT / 'model_random_forest_177.joblib')
    bundle = {'scaler': prep['scaler'], 'fitur_numerik_scaler': NUMERIC,
              'kmeans': stage1['kmeans'], 'invert_clusters': stage1['invert_clusters'],
              'features_kmeans': KM_FEATURES, 'rf_model': rf, 'features_rf': RF_FEATURES,
              'target_classes': rf.classes_.tolist(),
              'metadata': {'model_version': VERSION, 'pipeline_version': PIPELINE_VERSION,
                           'algorithm': 'K-Means 2D K=2 + Random Forest 100 Trees',
                           'dataset_count': len(df), 'training_count': len(train), 'test_count': len(test),
                           'preprocessing_fit': 'train_only', 'target_type': 'rule_based_proxy',
                           'cluster_label_note': LABEL_NOTE,
                           'centroids': geometry(prep['scaler'], stage1['kmeans'], stage1['invert_clusters'])}}
    joblib.dump(bundle, OUTPUT / 'pipeline_bundle_177.joblib')
    print(f'RF 100 pohon; latih {len(train)}, uji {len(test)}; akurasi uji {rf.score(test[RF_FEATURES], test[TARGET]):.4%}.')
    return bundle


def bundle_and_data():
    bundle = joblib.load(OUTPUT / 'pipeline_bundle_177.joblib')
    bundle['rf_model'].n_jobs = 1
    return bundle, pd.read_csv(BASE / 'dataset_terklaster_177.csv'), pd.read_csv(BASE / 'hasil_prediksi_rf_177.csv')


def evaluate():
    bundle, df, heldout = bundle_and_data()
    prep, rf = load_preprocessing(), bundle['rf_model']
    indexed = df.set_index('ID_Siswa')
    train, test = indexed.loc[prep['train_ids']], indexed.loc[prep['test_ids']]
    predictions = heldout.Prediksi_Random_Forest
    truth = heldout[TARGET]
    classes = rf.classes_.tolist()
    support = truth.value_counts().reindex(classes, fill_value=0)
    evaluated = [c for c in classes if support[c] > 0]
    km_labels = df.Klaster_KMeans.to_numpy()
    tr_labels = train.Klaster_KMeans.to_numpy()
    km_scores = {'k': 2, 'fit_rows': len(train), 'assignment_rows': len(df),
                 'cluster_counts': {str(k): int(v) for k, v in df.Klaster_KMeans.value_counts().items()},
                 'centroids': bundle['metadata']['centroids'],
                 'inertia_train': float(bundle['kmeans'].inertia_),
                 'silhouette_train': float(silhouette_score(train[KM_FEATURES], tr_labels)),
                 'silhouette_all_assigned': float(silhouette_score(df[KM_FEATURES], km_labels)),
                 'davies_bouldin_all_assigned': float(davies_bouldin_score(df[KM_FEATURES], km_labels))}
    report = classification_report(truth, predictions, labels=classes, output_dict=True, zero_division=0)
    scored = classification_report(truth, predictions, labels=evaluated, output_dict=True, zero_division=0)
    metrics = {'model_version': VERSION, 'pipeline_version': PIPELINE_VERSION, 'rows': len(df),
               'training_rows': len(train), 'test_rows': len(test), 'target_type': 'rule_based_proxy',
               'classes': classes, 'target_counts': {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
               'test_support': {str(k): int(v) for k, v in support.items()},
               'untested_classes': [c for c in classes if support[c] == 0],
               'train_accuracy': float(rf.score(train[RF_FEATURES], train[TARGET])),
               'test_accuracy': float(accuracy_score(truth, predictions)),
               'test_correct': int((truth == predictions).sum()),
               'classification_report_all_labels': report,
               'macro_evaluated_classes': scored['macro avg'],
               'confusion_matrix': confusion_matrix(truth, predictions, labels=classes).tolist(),
               'feature_importance': dict(zip(RF_FEATURES, rf.feature_importances_.tolist())),
               'confidence_min': float(heldout['Confidence_Score (%)'].min()),
               'confidence_mean': float(heldout['Confidence_Score (%)'].mean()),
               'confidence_below_50': int((heldout['Confidence_Score (%)'] < 50).sum()), 'kmeans': km_scores}
    engineered = pd.read_csv(BASE / 'dataset_engineered_177.csv')
    tr, te = split_indices(engineered)
    baseline = engineered.iloc[tr][TARGET].mode().iloc[0]
    metrics['baseline_majority_accuracy'] = float((engineered.iloc[te][TARGET] == baseline).mean())
    without = fit_hybrid_in_memory(engineered, tr, te, with_cluster=False)
    metrics['rf_without_cluster_accuracy'] = float(accuracy_score(engineered.iloc[te][TARGET], without['prediction']))
    folds = []
    with threadpool_limits(limits=1):
        for fold, (tr_fold, te_fold) in enumerate(RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42).split(engineered, engineered[TARGET]), 1):
            fitted = fit_hybrid_in_memory(engineered, tr_fold, te_fold)
            folds.append({'fold': fold, 'training_n': len(tr_fold), 'test_n': len(te_fold),
                          'accuracy': float(accuracy_score(engineered.iloc[te_fold][TARGET], fitted['prediction']))})
    metrics['repeated_2fold_5repeats'] = folds
    metrics['cv_accuracy_mean'] = float(np.mean([r['accuracy'] for r in folds]))
    metrics['cv_accuracy_std_descriptive'] = float(np.std([r['accuracy'] for r in folds]))
    write_json(OUTPUT / 'metrics_177.json', metrics)
    pd.DataFrame({'Fitur': RF_FEATURES, 'Gini_importance': rf.feature_importances_}).to_csv(OUTPUT / 'feature_importance_177.csv', index=False)
    print(f'Evaluasi: {metrics["test_correct"]}/{len(test)} benar; kelas belum diuji: {metrics["untested_classes"]}.')
    return metrics


def save_figure(fig, name):
    fig.savefig(OUTPUT / name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_kmeans():
    bundle, df, _ = bundle_and_data()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {0: '#cf791f', 1: '#21956c'}
    for code in [0, 1]:
        part = df[df.Klaster_KMeans == code]
        ax.scatter(part.Skor_Minat_Kejuruan, part.Rata_Rapor, color=colors[code], alpha=.65,
                   label=f'{CLUSTER_NAMES[code]} (n={len(part)})', s=35)
    for c in bundle['metadata']['centroids']:
        ax.scatter(c['x'], c['y'], marker='X', s=220, color=colors[c['kode']], edgecolor='black',
                   label=f'Centroid latih C{c["kode"]}: ({c["x"]:.2f}, {c["y"]:.2f})')
    ax.set(xlim=(7, 35), ylim=(0, 100), xlabel='X: skor minat kejuruan RIASEC (7–35)',
           ylabel='Y: rata-rata empat pelajaran (0–100)', title='K-Means 2D, K=2 · Model 1.1.0 · fit 141 data latih')
    ax.legend(fontsize=8, loc='lower left')
    fig.text(.5, -.02, 'Label operasional kelompok; bukan aturan dua ambang minimum atau diagnosis salah jurusan.', ha='center', fontsize=8)
    save_figure(fig, 'scatter_kmeans_2d_177.png')
    sil = silhouette_samples(df[KM_FEATURES], df.Klaster_KMeans)
    fig, ax = plt.subplots(figsize=(8, 5))
    lower = 5
    for code in [0, 1]:
        values = np.sort(sil[df.Klaster_KMeans == code])
        ax.fill_betweenx(np.arange(lower, lower + len(values)), 0, values, alpha=.7, color=colors[code],
                         label=f'C{code}: {CLUSTER_NAMES[code]} (n={len(values)})')
        lower += len(values) + 8
    ax.axvline(sil.mean(), color='#b23333', linestyle='--', label=f'Rerata seluruh assignment: {sil.mean():.4f}')
    ax.set(xlabel='Silhouette', yticks=[], title='Kualitas pengelompokan 177 siswa · centroid dipelajari dari 141 latih')
    ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(.5, -.3))
    save_figure(fig, 'evaluasi_silhouette_kmeans_177.png')


def plot_forest():
    bundle, _, heldout = bundle_and_data()
    rf = bundle['rf_model']; classes = rf.classes_.tolist()
    cm = confusion_matrix(heldout[TARGET], heldout.Prediksi_Random_Forest, labels=classes)
    accuracy = accuracy_score(heldout[TARGET], heldout.Prediksi_Random_Forest)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    labels = [SHORT_CLASSES[c] for c in classes]
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=labels, yticklabels=labels,
           xlabel='Prediksi', ylabel='Target proxy berbasis aturan', title=f'RF 100 pohon · akurasi {accuracy:.2%}\nSosial: support uji 0, kinerja belum terukur')
    fig.colorbar(im, ax=ax)
    save_figure(fig, 'confusion_matrix_rf_177.png')
    values = pd.Series(rf.feature_importances_, index=RF_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(values.index.str.replace('_scaled', '', regex=False), values.values, color='#326e9b')
    for i, value in enumerate(values): ax.text(value + .002, i, f'{value:.2%}', va='center', fontsize=8)
    ax.set(xlim=(0, values.max()*1.2), xlabel='Gini importance (bukan pengaruh kausal)', title='Kepentingan prediktif fitur RF')
    save_figure(fig, 'feature_importance_rf_177.png')


def plot_votes():
    bundle, df, heldout = bundle_and_data()
    rf = bundle['rf_model']; classes = rf.classes_.tolist()
    x = df.set_index('ID_Siswa').loc[heldout.ID_Siswa, RF_FEATURES].to_numpy()
    votes = np.zeros((len(heldout), len(classes)), dtype=int)
    for tree in rf.estimators_:
        pred = tree.predict(x).astype(int)
        votes[np.arange(len(x)), pred] += 1
    assert np.all(votes.sum(axis=1) == 100)
    table = pd.DataFrame(votes, columns=classes)
    for name in ['ID_Siswa', 'Status_Kesesuaian', 'Status_Prediksi']:
        table[name] = heldout[name].to_numpy()
    table['Prediksi'] = heldout.Prediksi_Random_Forest.to_numpy()
    table.to_csv(OUTPUT / 'rekapitulasi_voting_100_pohon_177.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [2, 1]})
    left = np.zeros(len(x))
    for j, c in enumerate(classes):
        axes[0].barh(range(len(x)), votes[:, j], left=left, label=SHORT_CLASSES[c]); left += votes[:, j]
    axes[0].set(yticks=range(len(x)), yticklabels=heldout.ID_Siswa, xlim=(0, 100),
                xlabel='Jumlah pohon yang memilih kelas (hard vote)', title='Voting aktual 100 pohon untuk 36 siswa uji')
    axes[0].tick_params(axis='y', labelsize=7); axes[0].legend(fontsize=8)
    confidence = heldout['Confidence_Score (%)']
    counts, _, bars = axes[1].hist(confidence, bins=np.arange(0, 101, 10), edgecolor='white', color='#308a80')
    assert int(counts.sum()) == len(heldout)
    for n, bar in zip(counts, bars):
        if n: axes[1].text(bar.get_x()+5, n+.1, str(int(n)), ha='center')
    axes[1].set(xlim=(0,100), xlabel='Confidence model (%)', ylabel='Jumlah siswa',
                title=f'Seluruh {len(heldout)} prediksi tercakup\n{int((confidence<50).sum())} confidence <50%')
    fig.text(.5,.02,'RF scikit-learn mengagregasi probabilitas daun; confidence belum dikalibrasi sebagai peluang keberhasilan kuliah.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1)); save_figure(fig, 'hasil_voting_nyata_100_pohon_177.png')


def plot_actual_tree():
    bundle, _, _ = bundle_and_data(); rf = bundle['rf_model']
    for name, depth in [('struktur_pohon_keputusan_asli_rf_177.png', 3), ('visualisasi_pohon_keputusan_rf_177.png', None)]:
        fig, ax = plt.subplots(figsize=(24, 11 if depth is not None else 17))
        plot_tree(rf.estimators_[0], max_depth=depth, feature_names=RF_FEATURES,
                  class_names=[SHORT_CLASSES[c] for c in rf.classes_], filled=True, rounded=True,
                  fontsize=7 if depth is not None else 5, ax=ax)
        ax.set_title(f'Tree #1 aktual · {"cuplikan kedalaman 3" if depth is not None else "pohon penuh"}\n'
                     'samples: jumlah sampel unik bootstrap; value: jumlah berbobot per kelas')
        save_figure(fig, name)


def plot_architecture():
    fig, ax = plt.subplots(figsize=(11, 7)); ax.axis('off')
    boxes = [('42 jawaban (6 skor komposit 7–35) + jurusan + 4 nilai rapor', .90),
             ('Scaler tersimpan: fit hanya 141 data latih', .73),
             ('Tahap 1: K-Means K=2 pada X=minat kejuruan dan Y=rerata rapor\nLabel operasional: Sesuai / Tidak Sesuai', .55),
             ('Tahap 2: RF 100 pohon\n10 fitur Z-score + 1 kode klaster biner', .36),
             ('Rumpun terprediksi + probabilitas + voting\nHubungan linier/lintas bidang ditentukan dari jurusan dan rumpun hasil', .15)]
    for text, y in boxes:
        ax.text(.5,y,text,ha='center',va='center',fontsize=11,
                bbox={'boxstyle':'round,pad=.7','facecolor':'#eaf2fa','edgecolor':'#39698e'})
    for (_,a),(_,b) in zip(boxes,boxes[1:]):
        ax.annotate('',xy=(.5,b+.07),xytext=(.5,a-.06),arrowprops={'arrowstyle':'->','lw':1.5})
    ax.set_title('Inferensi NALARA · model 1.1.0\nBootstrap dilakukan saat pelatihan RF; inferensi memakai estimator tersimpan')
    save_figure(fig,'diagram_arsitektur_random_forest.png')


def render_all():
    plot_kmeans(); plot_forest(); plot_votes(); plot_actual_tree(); plot_architecture()
    print('Grafik dihasilkan dari objek model dan metrik aktual.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        transform_dataset(); fit_kmeans(); fit_forest(); evaluate(); render_all()
