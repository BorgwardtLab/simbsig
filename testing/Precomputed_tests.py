# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

import unittest
import numpy as np
import pathlib
import h5py
import os

import scipy.spatial.distance

from sklearn.neighbors import NearestNeighbors as NearestNeighbors_sk,\
                              KNeighborsClassifier as KNeighborsClassifier_sk,\
                              KNeighborsRegressor as KNeighborsRegressor_sk,\
                              RadiusNeighborsRegressor as RadiusNeighborsRegressor_sk,\
                              RadiusNeighborsClassifier as RadiusNeighborsClassifier_sk

from simbsig.neighbors import NearestNeighbors
from simbsig.neighbors import KNeighborsClassifier
from simbsig.neighbors import KNeighborsRegressor
from simbsig.neighbors import RadiusNeighborsRegressor
from simbsig.neighbors import RadiusNeighborsClassifier

# Here are tests for the metric=='precomputed' functionality

class Test_metric_precomputed(unittest.TestCase):

    def test_metric_precomputed_nearest_neighbors_concept_empty_query(self):
        '''Test concept vs manually derived results'''

        # Test on a non-diagonal distance matrix. The results have been manually verified.
        # Here we test however that these manually verified results also match the sklearn output.
        X_s1 = [[0, 8, 7],
                [12, 0, 14],
                [20, 24, 0]]

        # 1 Neighbor, sorted
        nn_1 = NearestNeighbors_sk(n_neighbors=1, metric='precomputed')
        nn_1.fit(X_s1)
        res_sk_dist, res_sk_ind = nn_1.kneighbors()

        nn_2 = NearestNeighbors(n_neighbors=1, metric='precomputed')
        nn_2.fit(X_s1)
        res_simbsig_dist, res_simbsig_ind = nn_2.kneighbors(sort_results=True)

        self.assertTrue(np.array_equal(res_sk_dist, res_simbsig_dist))
        self.assertTrue(np.array_equal(res_sk_ind, res_simbsig_ind))

        # 2 Neighbors, sorted
        nn_1 = NearestNeighbors_sk(n_neighbors=2, metric='precomputed')
        nn_1.fit(X_s1)
        res_sk_dist, res_sk_ind = nn_1.kneighbors()

        nn_2 = NearestNeighbors(n_neighbors=2, metric='precomputed')
        nn_2.fit(X_s1)
        res_simbsig_dist, res_simbsig_ind = nn_2.kneighbors(sort_results=True)

        self.assertTrue(np.array_equal(res_sk_dist, res_simbsig_dist))
        self.assertTrue(np.array_equal(res_sk_ind, res_simbsig_ind))

        # 1 Neighbor, unsorted
        nn_1 = NearestNeighbors_sk(n_neighbors=1, metric='precomputed')
        nn_1.fit(X_s1)
        res_sk_dist, res_sk_ind = nn_1.kneighbors()

        nn_2 = NearestNeighbors(n_neighbors=1, metric='precomputed')
        nn_2.fit(X_s1)
        res_simbsig_dist, res_simbsig_ind = nn_2.kneighbors(sort_results=False)

        self.assertTrue(np.array_equal(res_sk_dist, res_simbsig_dist))
        self.assertTrue(np.array_equal(res_sk_ind, res_simbsig_ind))

        # 2 Neighbors, unsorted
        nn_1 = NearestNeighbors_sk(n_neighbors=2, metric='precomputed')
        nn_1.fit(X_s1)
        res_sk_dist, res_sk_ind = nn_1.kneighbors()

        nn_2 = NearestNeighbors(n_neighbors=2, metric='precomputed')
        nn_2.fit(X_s1)
        res_simbsig_dist, res_simbsig_ind = nn_2.kneighbors(sort_results=False)

        # without sorting, simbsig returns the neighbors just in the same order as they are in the training data
        self.assertTrue(np.array_equal([[1, 2],
                                        [0, 2],
                                        [0, 1]], res_simbsig_ind))

        self.assertTrue(np.array_equal([[8, 7],
                                        [12, 14],
                                        [20, 24]], res_simbsig_dist))

        # Check for different radius the radius_neighbors method
        for radius in [0, 1, 7.5, 13.5, 20.5]:
            nn_1 = NearestNeighbors_sk(radius=radius, metric='precomputed')
            nn_1.fit(X_s1)
            res_sk_dist, res_sk_ind = nn_1.radius_neighbors()

            nn_2 = NearestNeighbors(radius=radius, metric='precomputed')
            nn_2.fit(X_s1)
            res_simbsig_dist, res_simbsig_ind = nn_2.radius_neighbors(sort_results=True)

            # check for each query point if it is equal to sklearn
            for i in range(len(res_sk_ind)):
                self.assertTrue(np.array_equal(res_sk_ind[i], res_simbsig_ind[i]))
                self.assertTrue(np.array_equal(res_sk_dist[i], res_simbsig_dist[i]))

    def test_metric_precomputed_nearest_neighbors_concept_nonempty_query(self):
        '''Test concept vs manually derived results'''

        # Test on a non-diagonal distance matrix. The results have been manually verified.
        # Here we test however that these manually verified results also match the sklearn output.
        X_s1 = [[0, 8, 7],
                [12, 0, 14],
                [20, 24, 0]]

        X_query = [[17, 12, 9],
                   [12, 17, 6]]

        # 1 Neighbor, sorted
        nn_1 = NearestNeighbors_sk(n_neighbors=1, metric='precomputed')
        nn_1.fit(X_s1)
        res_sk_dist, res_sk_ind = nn_1.kneighbors(X_query)

        nn_2 = NearestNeighbors(n_neighbors=1, metric='precomputed',verbose=False)
        nn_2.fit(X_s1)
        res_simbsig_dist, res_simbsig_ind = nn_2.kneighbors(X_query, sort_results=True)

        self.assertTrue(np.array_equal(res_sk_dist, res_simbsig_dist))
        self.assertTrue(np.array_equal(res_sk_ind, res_simbsig_ind))

        # 2 Neighbors, sorted
        nn_1 = NearestNeighbors_sk(n_neighbors=2, metric='precomputed')
        nn_1.fit(X_s1)
        res_sk_dist, res_sk_ind = nn_1.kneighbors(X_query)

        nn_2 = NearestNeighbors(n_neighbors=2, metric='precomputed',verbose=False)
        nn_2.fit(X_s1)
        res_simbsig_dist, res_simbsig_ind = nn_2.kneighbors(X_query, sort_results=True)

        self.assertTrue(np.array_equal(res_sk_dist, res_simbsig_dist))
        self.assertTrue(np.array_equal(res_sk_ind, res_simbsig_ind))

        # 1 Neighbor, unsorted
        nn_1 = NearestNeighbors_sk(n_neighbors=1, metric='precomputed')
        nn_1.fit(X_s1)
        res_sk_dist, res_sk_ind = nn_1.kneighbors(X_query)

        nn_2 = NearestNeighbors(n_neighbors=1, metric='precomputed',verbose=False)
        nn_2.fit(X_s1)
        res_simbsig_dist, res_simbsig_ind = nn_2.kneighbors(X_query, sort_results=False)

        self.assertTrue(np.array_equal(res_sk_dist, res_simbsig_dist))
        self.assertTrue(np.array_equal(res_sk_ind, res_simbsig_ind))

        # 2 Neighbors, unsorted
        nn_1 = NearestNeighbors_sk(n_neighbors=2, metric='precomputed')
        nn_1.fit(X_s1)
        res_sk_dist, res_sk_ind = nn_1.kneighbors(X_query)

        nn_2 = NearestNeighbors(n_neighbors=2, metric='precomputed',verbose=False)
        nn_2.fit(X_s1)
        res_simbsig_dist, res_simbsig_ind = nn_2.kneighbors(X_query, sort_results=False)

        # without sorting, simbsig returns the neighbors just in the same order as they are in the training data
        # print(res_simbsig_ind)
        self.assertTrue(np.array_equal([[1, 2],
                                        [0, 2]], res_simbsig_ind))

        self.assertTrue(np.array_equal([[12, 9],
                                        [12, 6]], res_simbsig_dist))

        # Check for different radius the radius_neighbors method
        for radius in [0, 1, 7.5, 13.5, 20.5]:
            nn_1 = NearestNeighbors_sk(radius=radius, metric='precomputed')
            nn_1.fit(X_s1)
            res_sk_dist, res_sk_ind = nn_1.radius_neighbors()

            nn_2 = NearestNeighbors(radius=radius, metric='precomputed',verbose=False)
            nn_2.fit(X_s1)
            res_simbsig_dist, res_simbsig_ind = nn_2.radius_neighbors(sort_results=True)

            # check for each query point if it is equal to sklearn
            for i in range(len(res_sk_ind)):
                self.assertTrue(np.array_equal(res_sk_ind[i], res_simbsig_ind[i]))
                self.assertTrue(np.array_equal(res_sk_dist[i], res_simbsig_dist[i]))


    def test_metric_precomputed_kneighbors_classifier_concept_nonempty_query(self):
        '''Test concept vs manually derived results'''

        # Test on a non-diagonal distance matrix. The results have been manually verified.
        # Here we test however that these manually verified results also match the sklearn output.
        X_s1 = [[0, 8, 7],
                [12, 0, 14],
                [20, 24, 0]]
        y = [1,0,1]

        X_query = [[17, 12, 9],
                   [12, 17, 6]]

        # 1 Neighbor, sorted
        nn_1 = KNeighborsClassifier_sk(n_neighbors=1, metric='precomputed')
        nn_1.fit(X_s1, y)
        pred_sk = nn_1.predict(X_query)

        nn_2 = KNeighborsClassifier(n_neighbors=1, metric='precomputed',verbose=False)
        nn_2.fit(X_s1, y)
        pred_simbsig = nn_2.predict(X_query)

        self.assertTrue(np.array_equal(pred_sk, pred_simbsig))

        # 2 Neighbors, sorted
        nn_1 = KNeighborsClassifier_sk(n_neighbors=2, metric='precomputed')
        nn_1.fit(X_s1, y)
        pred_sk= nn_1.predict(X_query)

        nn_2 = KNeighborsClassifier(n_neighbors=2, metric='precomputed',verbose=False)
        nn_2.fit(X_s1, y)
        pred_simbsig = nn_2.predict(X_query)

        self.assertTrue(np.array_equal(pred_sk, pred_simbsig))

    def test_metric_precomputed_sklearn_simbsig(self):
        '''Compare simbsig metric_precomputedvs sklearn metric_precomputed as gold standard'''

        ################################################################################################################
        ################################################################################################################
        # Create Dataset
        np.random.seed(98)
        n_samples = 40
        n_queries = 15
        n_dim = 10 # Tested 1,2,3,4,5,10,15,20
        X_train = np.random.uniform(low=-5, high=5, size=(n_samples, n_dim))

        dist_mat = scipy.spatial.distance.cdist(X_train,X_train)

        X_query = np.random.uniform(low=0, high=10, size=(n_queries, n_samples))

        # Covariance Matrix and Inverted Covariance Matrix
        VI = np.linalg.inv(np.cov(X_train.T))

        # feature_weights vector
        w = np.random.uniform(size=(n_dim))

        ###############################################################################################################
        # Create integer labelled y_train_integer:
        y_train_integer = np.random.randint(low=0, high=5, size=(n_samples))

        ###############################################################################################################
        # with y_train_integer, bind this also to hdf5 dataset
        # Safe to hdf5 file format
        dataset_path = pathlib.Path(__file__).resolve().parents[0]
        train_file = f'train.hdf5'
        query_file = f'query.hdf5'

        with h5py.File(os.path.join(dataset_path, f"{train_file}"), 'w') as f:
             f.create_dataset("X",data=dist_mat)
             f.create_dataset("y",data=y_train_integer)

        with h5py.File(os.path.join(dataset_path, f"{query_file}"), 'w') as f:
             f.create_dataset("X",data=X_query)

        # Load hdf5 files
        train_data = h5py.File(os.path.join(dataset_path, train_file), 'r')
        query_data = h5py.File(os.path.join(dataset_path, query_file))

        ###############################################################################################################
        # Create categorical/string labelled datset:
        categorical_labels = np.array(['a','b','c','d','e'])
        a = np.random.randint(low=0, high=5, size=(n_samples), dtype=int)
        y_train_categorical = categorical_labels[a]

        ################################################################################################################
        # Loop through different combinations of arguments

        INF = 1000000 # sufficient for being larger than any number used during testing
        for n_neighbors in [1, 5, n_samples-1]:
            for radius in [1, 2, INF]: # radius 0 will make simbsig and sklearn raise warnings for the Radius... methods
                for device in ['cpu']:#,'gpu']:
                    for mode in ['hdf5', 'arrays']:
                        for batch_size in [1,10,INF]:
                            for sample_weights in ['uniform', 'distance']:
                                for sort_results in ['True']:

                                    inf_string = f'radius: {radius} sample_weights: {sample_weights} ' \
                                                 f'device: {device} mode: {mode} batch_size: {batch_size} ' \
                                    ####################################################################################
                                    # initiate
                                    nn_sk = NearestNeighbors_sk(n_neighbors=n_neighbors,
                                                                radius=radius,
                                                                metric='precomputed')
                                    nn_simbsig = NearestNeighbors(n_neighbors=n_neighbors,
                                                             radius=radius,
                                                             metric='precomputed',
                                                             device=device,
                                                             mode=mode,
                                                             batch_size=batch_size,
                                                             sort_results=sort_results,
                                                             verbose=False)

                                    kcl_sk = KNeighborsClassifier_sk(n_neighbors=n_neighbors,
                                                                     metric='precomputed',
                                                                     weights=sample_weights)
                                    kcl_simbsig = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                                  metric='precomputed',
                                                                  device=device,
                                                                  mode=mode,
                                                                  batch_size=batch_size,
                                                                  weights=sample_weights,
                                                                  verbose=False
                                                                  )

                                    kr_sk = KNeighborsRegressor_sk(n_neighbors=n_neighbors,
                                                                   metric='precomputed',
                                                                   weights=sample_weights)
                                    kr_simbsig = KNeighborsRegressor(n_neighbors=n_neighbors,
                                                                metric='precomputed',
                                                                device=device,
                                                                mode=mode,
                                                                batch_size=batch_size,
                                                                weights=sample_weights,
                                                                verbose=False
                                                                )

                                    rcl_sk = RadiusNeighborsClassifier_sk(radius=radius,
                                                                          metric='precomputed',
                                                                          outlier_label=-1,
                                                                          weights=sample_weights)
                                    rcl_simbsig = RadiusNeighborsClassifier(radius=radius,
                                                                       metric='precomputed',
                                                                       device=device,
                                                                       mode=mode,
                                                                       batch_size=batch_size,
                                                                       weights=sample_weights,
                                                                       outlier_label=-1,
                                                                       verbose=False
                                                                       )

                                    rr_sk = RadiusNeighborsRegressor_sk(radius=radius,
                                                                        metric='precomputed',
                                                                        weights=sample_weights)
                                    rr_simbsig = RadiusNeighborsRegressor(radius=radius,
                                                                     metric='precomputed',
                                                                     device=device,
                                                                     mode=mode,
                                                                     batch_size=batch_size,
                                                                     weights=sample_weights,
                                                                     verbose=False
                                                                     )

                                    ####################################################################################
                                    # fit

                                    # sklearn
                                    y_cl_sk = y_train_integer
                                    y_reg_sk = y_train_integer
                                    nn_sk.fit(dist_mat)
                                    kcl_sk.fit(dist_mat, y_cl_sk)
                                    kr_sk.fit(dist_mat, y_reg_sk)
                                    rcl_sk.fit(dist_mat, y_cl_sk)
                                    rr_sk.fit(dist_mat, y_reg_sk)

                                    # simbsig
                                    if mode=='arrays':
                                        y_cl_simbsig = y_train_integer
                                        y_reg_simbsig = y_train_integer
                                        nn_simbsig.fit(dist_mat)
                                        kcl_simbsig.fit(dist_mat, y_cl_simbsig)
                                        kr_simbsig.fit(dist_mat, y_reg_simbsig)
                                        rcl_simbsig.fit(dist_mat, y_cl_simbsig)
                                        rr_simbsig.fit(dist_mat, y_reg_simbsig)

                                    elif mode=='hdf5':
                                        nn_simbsig.fit(train_data)
                                        kcl_simbsig.fit(train_data)
                                        kr_simbsig.fit(train_data)
                                        rcl_simbsig.fit(train_data)
                                        rr_simbsig.fit(train_data)


                                    ####################################################################################
                                    # assert prediction

                                    ####################################################################################
                                    ####################################################################################
                                    # NearestNeighbors
                                    # kneighbors on training data
                                    dist_arr_sk, ind_arr_sk = nn_sk.kneighbors()
                                    dist_arr_simbsig, ind_arr_simbsig = nn_simbsig.kneighbors(sort_results=sort_results)

                                    self.assertTrue(np.allclose(dist_arr_sk, dist_arr_simbsig, rtol=1e-4))
                                    self.assertTrue(np.allclose(ind_arr_sk, ind_arr_simbsig))

                                    # kneighbors on query data
                                    dist_arr_sk_q, ind_arr_sk_q = nn_sk.kneighbors(X_query)
                                    if mode == 'arrays':
                                        dist_arr_simbsig_q, ind_arr_simbsig_q = nn_simbsig.kneighbors(X_query,
                                                                                       sort_results=sort_results)
                                    elif mode == 'hdf5':
                                        dist_arr_simbsig_q, ind_arr_simbsig_q = nn_simbsig.kneighbors(query_data,
                                                                                       sort_results=sort_results)

                                    self.assertTrue(np.allclose(dist_arr_sk_q, dist_arr_simbsig_q, rtol=1e-4))
                                    self.assertTrue(np.allclose(ind_arr_sk_q, ind_arr_simbsig_q))

                                    # radius_neighbors on training data
                                    dist_arr_sk, ind_arr_sk = nn_sk.radius_neighbors()
                                    dist_arr_simbsig, ind_arr_simbsig = nn_simbsig.radius_neighbors(sort_results=sort_results)

                                    # for radius neighbors, check each query point's neighbors
                                    # individually using np.allclose
                                    for i in range(n_samples):
                                        self.assertTrue(np.allclose(dist_arr_sk[i], dist_arr_simbsig[i], rtol=1e-4))
                                        self.assertTrue(np.allclose(ind_arr_sk[i], ind_arr_simbsig[i]))

                                    # radius_neighbors on query data
                                    dist_arr_sk_q, ind_arr_sk_q = nn_sk.radius_neighbors(X_query)
                                    if mode == 'arrays':
                                        dist_arr_simbsig_q, ind_arr_simbsig_q = nn_simbsig.radius_neighbors(X_query,
                                                                                         sort_results=sort_results)
                                    elif mode == 'hdf5':
                                        dist_arr_simbsig_q, ind_arr_simbsig_q = nn_simbsig.radius_neighbors(query_data,
                                                                                             sort_results=sort_results)

                                    # for radius neighbors, check each query point's neighbors
                                    # individually using np.allclose
                                    for i in range(n_queries):
                                        self.assertTrue(np.allclose(dist_arr_sk_q[i], dist_arr_simbsig_q[i], rtol=1e-4))
                                        self.assertTrue(np.allclose(ind_arr_sk_q[i], ind_arr_simbsig_q[i]))

                                    ####################################################################################
                                    ####################################################################################
                                    # KNeighborsClassifier
                                    # predict
                                    ypred_sk = kcl_sk.predict(X_query)
                                    if mode=='arrays':
                                        ypred_simbsig = kcl_simbsig.predict(X_query)
                                    elif mode == 'hdf5':
                                        ypred_simbsig = kcl_simbsig.predict(query_data)

                                    self.assertTrue(np.allclose(ypred_sk, ypred_simbsig), msg=inf_string)

                                    # predict_proba

                                    ypred_sk_proba = kcl_sk.predict_proba(X_query)
                                    if mode=='arrays':
                                        ypred_simbsig_proba = kcl_simbsig.predict_proba(X_query)
                                    elif mode == 'hdf5':
                                        ypred_simbsig_proba = kcl_simbsig.predict_proba(query_data)

                                    self.assertTrue(np.allclose(ypred_sk_proba, ypred_simbsig_proba), msg=inf_string)

                                    ####################################################################################
                                    ####################################################################################
                                    # KNeighborsRegressor
                                    ypred_sk = kr_sk.predict(X_query)
                                    if mode == 'arrays':
                                        ypred_simbsig = kr_simbsig.predict(X_query)
                                    elif mode == 'hdf5':
                                        ypred_simbsig = kr_simbsig.predict(query_data)

                                    self.assertTrue(np.allclose(ypred_sk, ypred_simbsig, rtol=1e-4), msg=inf_string)

                                    ####################################################################################
                                    ####################################################################################
                                    # RadiusNeighborsClassifier
                                    # predict
                                    ypred_sk = rcl_sk.predict(X_query)
                                    if mode == 'arrays':
                                        ypred_simbsig = rcl_simbsig.predict(X_query)
                                    elif mode == 'hdf5':
                                        ypred_simbsig = rcl_simbsig.predict(query_data)


                                    self.assertTrue(np.allclose(ypred_sk, ypred_simbsig), msg=inf_string)

                                    # predict_proba
                                    ypred_sk_proba = rcl_sk.predict(X_query)
                                    if mode == 'arrays':
                                        ypred_simbsig_proba = rcl_simbsig.predict(X_query)
                                    elif mode == 'hdf5':
                                        ypred_simbsig_proba = rcl_simbsig.predict(query_data)

                                    self.assertTrue(np.allclose(ypred_sk_proba, ypred_simbsig_proba), msg=inf_string)

                                    ####################################################################################
                                    ####################################################################################
                                    # RadiusNeighborsRegressor
                                    # Predict
                                    ypred_sk = rr_sk.predict(X_query)
                                    if mode == 'arrays':
                                        ypred_simbsig = rr_simbsig.predict(X_query)
                                    elif mode == 'hdf5':
                                        ypred_simbsig = rr_simbsig.predict(query_data)

                                    self.assertTrue(np.allclose(ypred_sk[i], ypred_simbsig[i], rtol=1e-4), msg=inf_string)

        train_data.close()
        query_data.close()

if __name__ == '__main__':
    unittest.main()
