"""Regresi: isolasi data uji, keselarasan bundle, dan geometri yang dikirim ke UI."""
import unittest
import warnings
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_pipeline import (BASE, OUTPUT, NUMERIC, KM_FEATURES, RF_FEATURES,
                               fit_hybrid_in_memory, split_indices)
from inference_engine import NalaraInferenceEngine


class ResearchRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        cls.df = pd.read_csv(BASE/'dataset_engineered_177.csv')
        cls.bundle = joblib.load(OUTPUT/'pipeline_bundle_177.joblib')

    def test_holdout_mutation_cannot_change_fitted_preprocessing_or_trees(self):
        train,test=split_indices(self.df)
        changed=self.df.copy()
        changed.loc[test,NUMERIC]=changed.loc[test,NUMERIC]*10+100
        with threadpool_limits(limits=1):
            a=fit_hybrid_in_memory(self.df,train,test)
            b=fit_hybrid_in_memory(changed,train,test)
        np.testing.assert_array_equal(a['scaler'].mean_,b['scaler'].mean_)
        np.testing.assert_array_equal(a['kmeans'].cluster_centers_,b['kmeans'].cluster_centers_)
        for ta,tb in zip(a['rf'].estimators_,b['rf'].estimators_):
            np.testing.assert_array_equal(ta.tree_.threshold,tb.tree_.threshold)

    def test_saved_transform_and_cluster_table_match_deployed_bundle(self):
        data=pd.read_csv(BASE/'dataset_terklaster_177.csv')
        self.assertEqual(int(self.bundle['scaler'].n_samples_seen_),141)
        np.testing.assert_allclose(self.bundle['scaler'].transform(data[NUMERIC]),
                                   data[[c+'_scaled' for c in NUMERIC]],atol=1e-12)
        with threadpool_limits(limits=1):
            labels=self.bundle['kmeans'].predict(data[KM_FEATURES])
        if self.bundle['invert_clusters']:labels=1-labels
        np.testing.assert_array_equal(labels,data.Klaster_KMeans)
        self.assertEqual(set(labels),{0,1})
        self.assertEqual(len(self.bundle['rf_model'].estimators_),100)

    def test_response_distances_and_relation_are_derived_from_real_models(self):
        engine=NalaraInferenceEngine()
        with threadpool_limits(limits=1):
            result=engine.predict_student('RPL',0,80,80,80,dict(zip(engine.RIASEC_DIMENSIONS,[24,31,20,23,22,25])))
        g=result['klaster_kmeans']['grafik_kmeans']
        self.assertEqual(g['y'],60)
        self.assertEqual(g['x'],31)
        expected=0 if g['jarak_centroid_0']<g['jarak_centroid_1'] else 1
        self.assertEqual(result['klaster_kmeans']['kode'],expected)
        self.assertEqual(result['hubungan_jurusan'],
                         'Linier' if result['rumpun_terprediksi']=='Sains & Teknologi Informasi' else 'Lintas bidang')
        self.assertEqual(sum(result['voting_pohon'].values()),100)
        self.assertEqual(result['model_metadata']['model_version'],'1.1.0')


if __name__=='__main__':
    unittest.main()
