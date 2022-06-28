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

from sklearn.neighbors import KNeighborsRegressor as KNeighborsRegressor_sk
from simbsig.neighbors import KNeighborsRegressor

# Here are tests for the KNeighborsRegressor implementation.

class Test_KNeighborsRegressor(unittest.TestCase):

    def test_KNeighborsRegressor_concept(self):
        '''Test concept vs manually derived results'''
        X_s1 = [[1]]
        y_s1 = [1]
        X_s2 = [[0], [1], [2], [3]]
        y_s2 = [1, 2, 3, 4]
        X_s3 = [[0, 0], [1, 0], [3, 4.5], [1, 1], [-2, -3]]
        y_s3 = [1, 2, 3, 4, 5]
        y_s4 = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]

        # Concept for kneighbors 1 training point of 1 dimension
        nn_1 = KNeighborsRegressor(n_neighbors=1)
        nn_1.fit(X_s1, y_s1)

        # Test Distance Matrix
        self.assertTrue(np.equal(np.array([1]), nn_1.kneighbors([[2]], sort_results=True))[0])
        # Test Index Matrix
        self.assertTrue(np.equal(np.array([0]), nn_1.kneighbors([[2]], sort_results=True))[1])
        # Test Regression
        self.assertTrue(np.equal(np.array(1), nn_1.predict([[2]])))

        # Concept for predict, 4 training points of 1 dimension
        nn_2 = KNeighborsRegressor(n_neighbors=3)
        nn_2.fit(X_s2, y_s2)
        # Test Regression
        self.assertTrue(np.allclose(np.array([2, 3]), nn_2.predict([[0.9], [2.1]])))

        # Concept for kneighbors 5 training points of 2 dimension incl negative values and string labels
        nn_3 = KNeighborsRegressor(n_neighbors=3)
        nn_3.fit(X_s3, y_s3)
        # Test Distance Matrix
        self.assertTrue(np.allclose(np.array([[1, 1, 1.41421356],
                                              [1, 2.82842712, 3.60555128]]),
                                    nn_3.kneighbors([[0, 1], [-2, -2]], sort_results=True)[0]))
        # Test Index Matrix
        self.assertTrue(np.allclose(np.array([[0, 3, 1],
                                              [4, 0, 1]]), nn_3.kneighbors([[0, 1], [-2, -2]], sort_results=True)[1]))
        # Test Regression
        self.assertTrue(np.allclose(np.array([7/3, 8/3]), nn_3.predict([[0, 1], [-2, -2]])))

        # Concept for kneighbors 5 training points of 2 dimension incl negative values and multiple classification
        nn_3 = KNeighborsRegressor(n_neighbors=3)
        nn_3.fit(X_s3, y_s4)
        # Test Distance Matrix
        self.assertTrue(np.allclose(np.array([[1, 1, 1.41421356],
                                              [1, 2.82842712, 3.60555128]]),
                                    nn_3.kneighbors([[0, 1], [-2, -2]], sort_results=True)[0]))
        # Test Index Matrix
        self.assertTrue(np.allclose(np.array([[0, 3, 1],
                                              [4, 0, 1]]), nn_3.kneighbors([[0, 1], [-2, -2]], sort_results=True)[1]))
        # Test Regression
        self.assertTrue(np.allclose(np.array([[7/3, 70/3],
                                                 [8/3, 80/3]]), nn_3.predict([[0, 1], [-2, -2]])))

class Test_KNeighborsRegressor_sklearn_simbsig(unittest.TestCase):
    def test_KNeighborsRegressor_sklearn_simbsig(self):
        '''Compare simbsig.neighbors.KNeighborsRegressor vs
        sklearn.neighbors.KNeighborsRegressor as gold standard'''

        ################################################################################################################
        # Create Dataset
        np.random.seed(98)
        n_samples = 100
        n_queries = 50
        n_dim = 10
        n_regressions = 2
        X_train = np.random.uniform(low=-5, high=5, size=(n_samples, n_dim))

        X_query = np.random.uniform(low=-5, high=5, size=(n_queries, n_dim))

        # Covariance Matrix and Inverted Covariance Matrix
        if n_dim > 1:
            VI = np.linalg.inv(np.cov(X_train.T))

        # feature_weights vector
        w = np.random.uniform(size=(n_dim))
        ###############################################################################################################
        # Create y_train:
        y_train = np.random.uniform(low=-5, high=5, size=(n_samples, n_regressions))

        # Safe to hdf5 file format
        dataset_path = pathlib.Path(__file__).resolve().parents[0]
        train_file = f'train.hdf5'
        query_file = f'query.hdf5'

        with h5py.File(os.path.join(dataset_path, f"{train_file}"), 'w') as f:
            f.create_dataset("X", data=X_train)
            f.create_dataset("y", data=y_train)

        with h5py.File(os.path.join(dataset_path, f"{query_file}"), 'w') as f:
            f.create_dataset("X", data=X_query)

        # Load hdf5 files
        train_data = h5py.File(os.path.join(dataset_path, train_file), 'r')
        query_data = h5py.File(os.path.join(dataset_path, query_file))

        ################################################################################################################
        # Loop through different combinations of arguments
        INF = 10000000  # sufficient as infinity for our tests
        for n_neighbors in [n_samples]:
            # if n_dim == 1, omit mahalanobis distance (depends on covariance matrix) and omit
            # cosine distance (is either 0 or 2 for points of 1 dimension)
            if n_dim == 1:
                metric_lys = ["minkowski", "euclidean", "manhattan"]
            else:
                metric_lys = ["euclidean",  "cosine", "mahalanobis", "minkowski", "manhattan"]
            for metric in metric_lys:
                for feature_weights in [None, w]:
                  for device in ['cpu']:#,'gpu']:
                        for mode in ['arrays', 'hdf5']:
                            for batch_size in [30, INF]:
                                for sample_weights in ['uniform', 'distance']:
                                    ####################################################################################
                                    # Initialize settings for this innermost loop
                                    # set default p
                                    p = 1.5

                                    # information string for error message
                                    inf_string = f'n_neighbors: {n_neighbors} sample_weights: {sample_weights}' \
                                                 f'metric: {metric} ' \
                                                 f'(if minkowski: p={p}) feature_weights: {feature_weights is not None} ' \
                                                 f'device: {device} mode: {mode} batch_size: {batch_size} '

                                    # prepare train and query data for sklearn if we have feature_weights
                                    if feature_weights is None:
                                        X_train_sklearn = X_train
                                        X_query_sklearn = X_query
                                    else:
                                        if metric == 'minkowski':
                                            feature_weights_actual = np.power(feature_weights, 1 / p)
                                        elif metric == 'manhattan' or metric == 'mahalanobis':
                                            feature_weights_actual = feature_weights
                                        elif metric == 'euclidean' or metric == 'cosine':
                                            feature_weights_actual = feature_weights ** 0.5

                                        X_train_sklearn = X_train * feature_weights_actual
                                        X_query_sklearn = X_query * feature_weights_actual

                                    # choose train and query data for simbsig based on mode
                                    if mode == 'arrays':
                                        X_train_used = X_train  # np.array
                                        y_train_used = y_train  # np.array
                                        X_query_used = X_query  # np.array
                                    elif mode == 'hdf5':
                                        X_train_used = train_data  # h5py.File
                                        X_query_used = query_data  # h5py.File

                                    ####################################################################################
                                    # create NearestNeighbor objects
                                    if metric == "mahalanobis":
                                        nn_sk = KNeighborsRegressor_sk(n_neighbors=n_neighbors, weights=sample_weights,
                                                                       metric=metric,
                                                                       metric_params={'VI': VI})
                                        nn_simbsig = KNeighborsRegressor(n_neighbors=n_neighbors, weights=sample_weights,
                                                                    metric=metric,
                                                                    feature_weights=feature_weights,
                                                                    device=device, mode=mode, batch_size=batch_size,
                                                                    metric_params={'VI': VI},
                                                                    verbose=False)
                                    else:
                                        nn_sk = KNeighborsRegressor_sk(n_neighbors=n_neighbors, weights=sample_weights,
                                                                       metric=metric,
                                                                       p=p)
                                        nn_simbsig = KNeighborsRegressor(n_neighbors=n_neighbors, weights=sample_weights,
                                                                    metric=metric,
                                                                    feature_weights=feature_weights, device=device,
                                                                    mode=mode,
                                                                    batch_size=batch_size, p=p,
                                                                    verbose=False)

                                    ####################################################################################
                                    # fit KNeighborsRegressorClassifier objects
                                    nn_sk.fit(X_train_sklearn, y_train)
                                    if mode == 'arrays':
                                        nn_simbsig.fit(X_train, y_train)
                                    elif mode == 'hdf5':
                                        nn_simbsig.fit(X_train_used)

                                    ####################################################################################
                                    # test predict
                                    y_pred_sk = nn_sk.predict(X_query_sklearn)
                                    y_pred_simbsig = nn_simbsig.predict(X_query_used)

                                    # Have tolerance for different prediction compared to sklearn due to
                                    # small, numerically caused distance differences
                                    acceptance_level = 0.001
                                    different_preds = np.sum(abs(y_pred_sk - y_pred_simbsig) > acceptance_level)
                                    same_preds = np.sum(abs(y_pred_sk - y_pred_simbsig) < acceptance_level)

                                    # The assertTrue statement is executed if there is a match (passed),
                                    # or more than 5% of all predicted targets do not match
                                    if different_preds == 0 or different_preds > n_queries*n_regressions*0.05:
                                        self.assertTrue(np.allclose(y_pred_sk, y_pred_simbsig, rtol=0.01),
                                                        msg=f'{same_preds}, {different_preds} {inf_string}')
                                    else:
                                        print(
                                            f'{inf_string} produced {different_preds} predictions'
                                            f' with difference > {acceptance_level}, (while {same_preds} '
                                            f' targets match below {acceptance_level} difference)')

        train_data.close()
        query_data.close()

if __name__ == '__main__':
    unittest.main()
