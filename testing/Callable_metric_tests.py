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

import sklearn.metrics
import torch
import math

from sklearn.metrics.pairwise import rbf_kernel

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

# Here are tests for the callable metric functionality

class Test_metric_callable(unittest.TestCase):

    def test_metric_callable_sklearn_simbsig(self):
        '''Compare simbsig callable distance metric vs sklearn callable distance metric as gold standard'''

        def rbf_metric(x1, x2, p=None, feature_weights=None, sigma=None):
            """Generic pairwise distance function
            Parameters:

            :parameter x1: torch.tensor of dimension (n_samples, n_features)
            :parameter x2: torch.tensor of dimension (m_samples, n_features)
            :parameter feature_weights: torch.tensor of dimension (n_features,)
            :parameter sigma: passed as metric_params={'sigma':int} in constructor. any custom parameter name may be
            chosen.

            Returns:

            :return dist_mat: numpy.array of dimension (n_samples, m_samples)

            Notice that n_samples does not have to be equal to m_samples. However, both n_features have to match.
            If GPU is available and a simbsig neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
            RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
            be handed over to custom_metric the GPU.

            """

            # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
            # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
            # np.array operations is possible. Optionally, feature weights can be used.

            # First step: compute pairwise euclidean distances
            euclidean_dist_mat = torch.pow(torch.cdist(x1, x2, 2), 2)

            # Second step: exp(-euclidean_distance/sigma)
            rbf_pairwise = torch.exp(-euclidean_dist_mat / sigma)
            # dist_mat = 1 - rbf_pairwise
            dist_mat = 1 - rbf_pairwise

            # 2. Move the result off of the tensor, and convert to numpy.array
            dist_mat = dist_mat.cpu().numpy()

            # 3. return the dist_mat
            return dist_mat

        ### An rbf kernel custom metric for sklearn's callable functionality
        def custom_metric(X1, X2):
            sigma=8
            distance = 0.0
            for i in range(len(X1)):
                distance += (X1[i]-X2[i])**2
            rbf = np.exp(-distance/sigma)
            rbf_similarity = 1- rbf
            return rbf_similarity

        ################################################################################################################
        ################################################################################################################
        # Create Dataset
        np.random.seed(98)
        n_samples = 40
        n_queries = 15
        n_dim = 2
        X_train = np.random.uniform(low=-5, high=5, size=(n_samples, n_dim))

        # dist_mat = scipy.spatial.distance.cdist(X_train,X_train)

        X_query = np.random.uniform(low=-5, high=5, size=(n_queries, n_dim))

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
             f.create_dataset("X",data=X_train)
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
                        for batch_size in [INF,1,10,INF]:
                            for sample_weights in ['uniform']:
                                for sort_results in ['True']:

                                    inf_string = f'radius: {radius} sample_weights: {sample_weights} ' \
                                                 f'device: {device} mode: {mode} batch_size: {batch_size} ' \
                                    ####################################################################################
                                    # initiate
                                    nn_sk = NearestNeighbors_sk(n_neighbors=n_neighbors,
                                                                radius=radius,
                                                                metric=custom_metric,
                                                                algorithm='brute')
                                    nn_simbsig = NearestNeighbors(n_neighbors=n_neighbors,
                                                             radius=radius,
                                                             metric=rbf_metric,
                                                             device=device,
                                                             mode=mode,
                                                             batch_size=batch_size,
                                                             sort_results=sort_results,
                                                             verbose=False,
                                                             metric_params={'sigma':8})

                                    kcl_sk = KNeighborsClassifier_sk(n_neighbors=n_neighbors,
                                                                     metric=custom_metric,
                                                                     weights=sample_weights,
                                                                     algorithm='brute')
                                    kcl_simbsig = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                                  metric=rbf_metric,
                                                                  device=device,
                                                                  mode=mode,
                                                                  batch_size=batch_size,
                                                                  weights=sample_weights,
                                                                  verbose=False,
                                                                  metric_params={'sigma': 8}
                                                                  )

                                    kr_sk = KNeighborsRegressor_sk(n_neighbors=n_neighbors,
                                                                   metric=custom_metric,
                                                                   weights=sample_weights,
                                                                   algorithm='brute')
                                    kr_simbsig = KNeighborsRegressor(n_neighbors=n_neighbors,
                                                                metric=rbf_metric,
                                                                device=device,
                                                                mode=mode,
                                                                batch_size=batch_size,
                                                                weights=sample_weights,
                                                                verbose=False,
                                                                metric_params={'sigma': 8}
                                                                )

                                    rcl_sk = RadiusNeighborsClassifier_sk(radius=radius,
                                                                          metric=custom_metric,
                                                                          outlier_label=-1,
                                                                          weights=sample_weights,
                                                                          algorithm='brute')
                                    rcl_simbsig = RadiusNeighborsClassifier(radius=radius,
                                                                       metric=rbf_metric,
                                                                       device=device,
                                                                       mode=mode,
                                                                       batch_size=batch_size,
                                                                       weights=sample_weights,
                                                                       outlier_label=-1,
                                                                       verbose=False,
                                                                       metric_params={'sigma': 8}
                                                                       )

                                    rr_sk = RadiusNeighborsRegressor_sk(radius=radius,
                                                                        metric=custom_metric,
                                                                        weights=sample_weights,
                                                                        algorithm='brute')
                                    rr_simbsig = RadiusNeighborsRegressor(radius=radius,
                                                                     metric=rbf_metric,
                                                                     device=device,
                                                                     mode=mode,
                                                                     batch_size=batch_size,
                                                                     weights=sample_weights,
                                                                     verbose=False,
                                                                     metric_params={'sigma': 8}
                                                                     )

                                    ####################################################################################
                                    # fit

                                    # sklearn
                                    y_cl_sk = y_train_integer
                                    y_reg_sk = y_train_integer
                                    nn_sk.fit(X_train)
                                    kcl_sk.fit(X_train, y_cl_sk)
                                    kr_sk.fit(X_train, y_reg_sk)
                                    rcl_sk.fit(X_train, y_cl_sk)
                                    rr_sk.fit(X_train, y_reg_sk)

                                    # simbsig
                                    if mode=='arrays':
                                        y_cl_simbsig = y_train_integer
                                        y_reg_simbsig = y_train_integer
                                        nn_simbsig.fit(X_train)
                                        kcl_simbsig.fit(X_train, y_cl_simbsig)
                                        kr_simbsig.fit(X_train, y_reg_simbsig)
                                        rcl_simbsig.fit(X_train, y_cl_simbsig)
                                        rr_simbsig.fit(X_train, y_reg_simbsig)

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
                                        ind_sort_sk = np.argsort(dist_arr_sk_q[i])
                                        self.assertTrue(np.allclose(dist_arr_sk_q[i][ind_sort_sk],
                                                                    dist_arr_simbsig_q[i], rtol=1e-4))
                                        self.assertTrue(np.array_equal(ind_arr_sk_q[i][ind_sort_sk],
                                                                       ind_arr_simbsig_q[i]))
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
