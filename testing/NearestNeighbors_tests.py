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
import torch
from sklearn.metrics.pairwise import rbf_kernel
import scipy.spatial.distance
from sklearn.neighbors import NearestNeighbors as NearestNeighbors_sk

from simbsig.neighbors import NearestNeighbors

# Here are tests for the NearestNeighbors implementation.

class Test_NearestNeighbors(unittest.TestCase):

    def test_NearestNeighbors_edgecases(self):
        '''Test edgecases vs manually derived results'''
        X_s1 = [[1]]
        X_s2 = [[0], [1], [2], [3]]

        # Throw Error if ask for more neighbors than samples in training set
        nn_1 = NearestNeighbors(n_neighbors=2)
        nn_1.fit(X_s1)
        try:
            nn_1.kneighbors([[1]])
        except ValueError:
            pass
        else:
            raise AssertionError("k > n failed to raise ValueError")

        # Return empty array if no neighbor in radius for a single point
        nn_2 = NearestNeighbors(radius=0.3)
        nn_2.fit(X_s2)
        ref = np.empty(1, dtype=object)
        ref[0] = np.array([])
        self.assertEqual(ref.shape, nn_2.radius_neighbors([[0.5]])[0].shape)
        self.assertEqual(ref.shape, nn_2.radius_neighbors([[0.5]])[1].shape)

        # Throw NO error if radius_neighbors with radius 0: check that result is correct
        nn_3 = NearestNeighbors(radius=0, n_neighbors=1)
        nn_3.fit(X_s2)
        nn_3_sk = NearestNeighbors_sk(radius=0, n_neighbors=1)
        nn_3_sk.fit(X_s2)

        # Test single nearest neighbor distance
        self.assertTrue(np.allclose(nn_3_sk.radius_neighbors(X_s1)[0][0], nn_3.radius_neighbors(X_s1)[0][0]))
        # Test single nearest neighbor index
        self.assertTrue(np.allclose(nn_3_sk.radius_neighbors(X_s1)[1][0], nn_3.radius_neighbors(X_s1)[1][0]))

        # Throw error if radius_neighbors with radius negative
        nn_4 = NearestNeighbors(radius=-1)
        nn_4.fit(X_s2)
        try:
            result = nn_4.radius_neighbors(X_s1)
        except ValueError:
            pass
        else:
            raise AssertionError("Failed to raise ValueError if radius < 0")

        # Raise error if radius_neighbors without radius
        nn_5 = NearestNeighbors(n_neighbors=4)
        nn_5.fit(X_s2)
        try:
            result = nn_5.radius_neighbors(X_s1)
        except ValueError:
            pass
        else:
            raise AssertionError("Failed to raise ValueError if radius is None")

        # If hand over radius and ask for kneighbors, do kneighbors "ordinary"
        nn_6 = NearestNeighbors(radius=0.9, n_neighbors=1)
        nn_6.fit(X_s1)
        nn_6_onlyNeighbors = NearestNeighbors(n_neighbors=1)
        nn_6_onlyNeighbors.fit(X_s1)

        self.assertTrue(np.allclose(nn_6_onlyNeighbors.kneighbors(X_s2), nn_6.kneighbors(X_s2)))
        self.assertEqual(nn_6_onlyNeighbors.kneighbors(X_s2)[0].shape, nn_6.kneighbors(X_s2)[0].shape)
        self.assertEqual(nn_6_onlyNeighbors.kneighbors(X_s2)[1].shape, nn_6.kneighbors(X_s2)[1].shape)
        self.assertEqual(nn_6_onlyNeighbors.kneighbors(X_s2)[0].dtype, nn_6.kneighbors(X_s2)[0].dtype)
        self.assertEqual(nn_6_onlyNeighbors.kneighbors(X_s2)[1].dtype, nn_6.kneighbors(X_s2)[1].dtype)

    def test_NearestNeighbors_concept(self):
        '''Test concept vs manually derived results'''
        X_s1 = [[1]]
        X_s2 = [[0], [1], [2], [3]]
        X_s3 = [[0, 0], [1, 0], [3, 4.5], [1, 1.25], [-2, -3]]

        # Concept for kneighbors 1 training point of 1 dimension
        nn_1 = NearestNeighbors(n_neighbors=1)
        nn_1.fit(X_s1)
        # Test Distance Matrix
        self.assertTrue(np.equal(np.array([1]), nn_1.kneighbors([[2]]))[0])
        # Test Index Matrix
        self.assertTrue(np.equal(np.array([0]), nn_1.kneighbors([[2]]))[1])

        # Concept for kneighbors 4 training points of 1 dimension
        nn_2 = NearestNeighbors(n_neighbors=2)
        nn_2.fit(X_s2)

        # Test Distance Matrix
        self.assertTrue(np.allclose(np.array([[0.1, 0.9], [0.1, 0.9]]), nn_2.kneighbors([[1.1], [2.1]])[0]))
        # Test Index Matrix
        self.assertTrue(np.allclose(np.array([[1, 2], [2, 3]]), nn_2.kneighbors([[1.1], [2.1]])[1]))

        # Concept for kneighbors 5 training points of 2 dimension incl negative values
        nn_3 = NearestNeighbors(n_neighbors=2)
        nn_3.fit(X_s3)
        # Test Distance Matrix
        self.assertTrue(np.allclose(np.array([[1.75, 2.5], [1, np.sqrt(8)]]), nn_3.kneighbors([[1, 3], [-2, -2]])[0]))
        # Test Index Matrix
        self.assertTrue(np.allclose(np.array([[3, 2], [4, 0]]), nn_3.kneighbors([[1, 3], [-2, -2]])[1]))

        ### Concept for radius_neighbor including a query point with no neighbors in training set
        nn_4 = NearestNeighbors(radius=0.6)
        nn_4.fit(X_s2)

        ref = np.empty(2, dtype=object)
        ref[0] = np.array([[0.5, 0.5]])
        ref[1] = np.array([[]])
        ind = np.empty(2, dtype=object)
        ind[0] = np.array([0, 1])
        ind[1] = np.array([])
        # Assert same output shape, which is (2,)
        self.assertEqual(ref.shape, nn_4.radius_neighbors([[0.5], [8]])[0].shape)
        # Assert distances of query point 1
        self.assertTrue(np.allclose(ref[0], nn_4.radius_neighbors([[0.5], [8]])[0][0]))
        # Assert distances of query point 2
        self.assertTrue(np.allclose(ref[1], nn_4.radius_neighbors([[0.5], [8]])[0][1]))
        # Assert indices of query point 1
        self.assertTrue(np.allclose(ind[0], nn_4.radius_neighbors([[0.5], [8]])[1][0]))
        # Assert indices of query point 2
        self.assertTrue(np.allclose(ind[1], nn_4.radius_neighbors([[0.5], [8]])[1][1]))

class Test_NearestNeighbors_sklearn_simbsig(unittest.TestCase):

    def test_NearestNeighbors_sklearn_simbsig(self):
        '''Compare simbsig.MiniBatchKMeans vs sklearn.clustering.KMeans as gold standard'''

        ################################################################################################################
        # Create Dataset
        np.random.seed(98)
        n_samples = 100
        n_queries = 50
        n_dim = 10
        X_train = np.random.uniform(low=-5,high=5, size=(n_samples, n_dim))
        y_train = np.random.uniform(low=-5,high=5, size=(n_samples))
        X_query = np.random.uniform(low=-5,high=5, size=(n_queries, n_dim))

        # Covariance Matrix and Inverted Covariance Matrix
        VI = np.linalg.inv(np.cov(X_train.T))

        # feature_weights vector
        w = np.random.uniform(size=(n_dim))

        # Safe to hdf5 file format
        dataset_path = pathlib.Path(__file__).resolve().parents[0]
        train_file = f'train.hdf5'
        query_file = f'query.hdf5'

        with h5py.File(os.path.join(dataset_path, f"{train_file}"), 'w') as f:
             f.create_dataset("X",data=X_train)
             f.create_dataset("y",data=y_train)

        with h5py.File(os.path.join(dataset_path, f"{query_file}"), 'w') as f:
             f.create_dataset("X",data=X_query)

        # Load hdf5 files
        train_data = h5py.File(os.path.join(dataset_path, train_file), 'r')
        query_data = h5py.File(os.path.join(dataset_path, query_file))

        ################################################################################################################
        # Loop through different combinations of arguments
        INF = 10000000 # sufficient as infinity for our tests
        for n_neighbors in [5, n_samples]:
            for radius in [None, 2, INF]:
                if n_dim == 1:
                    # if n_dim == 1, omit mahalanobis distance (depends on covariance matrix) and omit
                    # cosine distance (is either 0 or 2 for points of 1 dimension)
                    metric_lys = ["euclidean", "minkowski", "manhattan"]
                else:
                    metric_lys = ["mahalanobis", "euclidean", "minkowski", "manhattan", "cosine"]
                for metric in metric_lys:
                    for feature_weights in [w, None]:
                        for device in ['cpu']:#,'gpu']:
                            for mode in ['arrays', 'hdf5']:
                                for batch_size in [30, INF]:
                                    for sort_results in [True, False]:

                                        ####################################################################################
                                        # Initialize settings for this innermost loop
                                        # set default p
                                        p = 1.5

                                        # tolerated relative difference to sklearn's distances, in the np.allclose function
                                        # is more loose for expected smaller absolute distances
                                        rel_diff = 1e-6 # this is the np.allclose default
                                        if n_dim < 4:
                                            rel_diff = 1e-3
                                        elif n_dim < 8:
                                            rel_diff = 1e-5

                                        # information string for error message
                                        inf_string = f'n_neighbors: {n_neighbors} radius: {radius} metric: {metric} ' \
                                                     f'(if minkowski: p={p})' \
                                                     f'feature_weights: {feature_weights is not None} ' \
                                                     f'device: {device} mode: {mode} batch_size: {batch_size} ' \
                                                     f'sort_results: {sort_results}'

                                        # prepare train and query data for sklearn if we have feature_weights
                                        if feature_weights is None:
                                            X_train_sklearn = X_train
                                            X_query_sklearn = X_query
                                        else:
                                            if metric=='minkowski':
                                                feature_weights_actual = np.power(feature_weights, 1/p)
                                            elif metric=='manhattan' or metric=='mahalanobis':
                                                feature_weights_actual = feature_weights
                                            elif metric=='euclidean' or metric=='cosine':
                                                feature_weights_actual = feature_weights**0.5

                                            X_train_sklearn = X_train * feature_weights_actual
                                            X_query_sklearn = X_query * feature_weights_actual

                                        # choose train and query data for simbsig based on mode
                                        if mode == 'arrays':
                                            X_train_used = X_train # np.array
                                            X_query_used = X_query # np.array
                                        elif mode == 'hdf5':
                                            X_train_used = train_data # h5py.File
                                            X_query_used = query_data # h5py.File

                                        ####################################################################################
                                        # create NearestNeighbor objects
                                        if metric == "mahalanobis":
                                            nn_sk = NearestNeighbors_sk(n_neighbors=n_neighbors, radius=radius,
                                                                        metric=metric, metric_params={'VI':VI})
                                            nn_simbsig = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, metric=metric,
                                                                     feature_weights=feature_weights,
                                                                     device=device, mode=mode, batch_size=batch_size,
                                                                     metric_params={'VI':VI},
                                                                     verbose=False)
                                        else:
                                            nn_sk = NearestNeighbors_sk(n_neighbors=n_neighbors, radius=radius,
                                                                        metric=metric, p=p)
                                            nn_simbsig = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, metric=metric,
                                                                     feature_weights=feature_weights, device=device,
                                                                     mode=mode, batch_size=batch_size, p=p,
                                                                     verbose=False)

                                        ####################################################################################
                                        # fit NearestNeighbor objects
                                        nn_sk.fit(X_train_sklearn)
                                        if mode == 'arrays':
                                            nn_simbsig.fit(X_train)
                                        elif mode == 'hdf5':
                                            nn_simbsig.fit(X_train_used)

                                        ####################################################################################
                                        # Assert kneighbors
                                        if radius is None:
                                            # Note that sklearn always sorts results
                                            dist_arr_sk_sorted, ind_arr_sk_sorted = nn_sk.kneighbors(X_query_sklearn)
                                            dist_arr_simbsig, ind_arr_simbsig = nn_simbsig.kneighbors(X_query_used,
                                                                                             sort_results=sort_results)

                                            if sort_results:
                                                dist_arr_simbsig_sorted = dist_arr_simbsig
                                                ind_arr_simbsig_sorted = ind_arr_simbsig
                                            else:
                                                sort_idxs_simbsig = np.argsort(dist_arr_simbsig, axis=1)

                                                dist_arr_simbsig_sorted = np.take_along_axis(dist_arr_simbsig, sort_idxs_simbsig, axis=1)
                                                ind_arr_simbsig_sorted = np.take_along_axis(ind_arr_simbsig, sort_idxs_simbsig, axis=1)

                                            if (metric == "cosine" and n_dim < 15) or n_dim == 1:
                                                self.assertTrue(np.allclose(dist_arr_sk_sorted[:,3:],
                                                                            dist_arr_simbsig_sorted[:,3:], rtol=rel_diff),

                                                                msg=inf_string)
                                            else:
                                                self.assertTrue(np.allclose(dist_arr_sk_sorted,
                                                                            dist_arr_simbsig_sorted, rtol=rel_diff),
                                                                msg=inf_string)

                                            # Have tolerance for different indexing compared to sklearn due to
                                            # small, numerically caused distance differences
                                            different_indexes = np.sum(ind_arr_sk_sorted != ind_arr_simbsig_sorted)
                                            same_indexes = np.sum(ind_arr_sk_sorted == ind_arr_simbsig_sorted)

                                            # The assertTrue statement is executed if there is a perfect match (passed),
                                            # or more than 5% of all neighbors do not match (fail) (there are n_neighbors for each
                                            # of the n_queries
                                            if different_indexes == 0 or different_indexes > (n_queries*n_neighbors)/ 20:
                                                self.assertTrue(np.allclose(ind_arr_sk_sorted,
                                                                            ind_arr_simbsig_sorted),
                                                                msg=inf_string)
                                            else:
                                                print(f'{inf_string} produced {different_indexes} different'
                                                      f' indices (while {same_indexes} indices match)')

                                        ####################################################################################
                                        # Assert RadiusNeighbors
                                        # In radius_neighbors, have to test each array individually:
                                        # one such array corresponds to 1 point in the query set, and contains information
                                        # about this query point's neighbors from the train set within radius.
                                        else:
                                            dist_arr_sk, ind_arr_sk = nn_sk.radius_neighbors(X_query_sklearn,
                                                                                             sort_results=sort_results)
                                            dist_arr_simbsig, ind_arr_simbsig = nn_simbsig.radius_neighbors(X_query_used,
                                                                                             sort_results=sort_results)
                                            for query in range(n_queries):
                                                if sort_results:
                                                    dist_arr_sk_sorted = dist_arr_sk[query]
                                                    dist_arr_simbsig_sorted = dist_arr_simbsig[query]
                                                    ind_arr_sk_sorted = ind_arr_sk[query]
                                                    ind_arr_simbsig_sorted = ind_arr_simbsig[query]
                                                else:
                                                    sort_idxs_sk = np.argsort(dist_arr_sk[query])
                                                    sort_idxs_simbsig = np.argsort(dist_arr_simbsig[query])

                                                    dist_arr_sk_sorted = dist_arr_sk[query][sort_idxs_sk]
                                                    dist_arr_simbsig_sorted = dist_arr_simbsig[query][sort_idxs_simbsig]

                                                    ind_arr_sk_sorted = ind_arr_sk[query][sort_idxs_sk]
                                                    ind_arr_simbsig_sorted = ind_arr_simbsig[query][sort_idxs_simbsig]

                                                if metric == "cosine" and n_dim < 15 or n_dim == 1:
                                                    self.assertTrue(np.allclose(dist_arr_sk_sorted[3:],
                                                                                    dist_arr_simbsig_sorted[3:], rtol=rel_diff),
                                                                    msg=inf_string)
                                                else:
                                                    self.assertTrue(np.allclose(dist_arr_sk_sorted,
                                                                                    dist_arr_simbsig_sorted, rtol=rel_diff),
                                                                    msg=inf_string)

                                                # Have tolerance for different indexing compared to sklearn due to
                                                # small, numerically caused distance differences
                                                different_indexes = np.sum(ind_arr_sk_sorted != ind_arr_simbsig_sorted)
                                                same_indexes = np.sum(ind_arr_sk_sorted == ind_arr_simbsig_sorted)
                                                # The assertTrue statement is executed if there is a perfect match (passed),
                                                # or more than 5% of the neighbors do not match (fail) (there
                                                # are n_neighbors many neighbors considered at once here)
                                                if different_indexes == 0 or different_indexes > n_neighbors/20:
                                                    self.assertTrue(np.allclose(ind_arr_sk_sorted,
                                                                                ind_arr_simbsig_sorted),
                                                                    msg=inf_string)
                                                else:
                                                    print(f'{inf_string} produced {different_indexes} different'
                                                          f' indices (while {same_indexes} indices match)')

        train_data.close()
        query_data.close()

if __name__ == '__main__':
    unittest.main()
