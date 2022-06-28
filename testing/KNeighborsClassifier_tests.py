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

from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_sk
from simbsig.neighbors import KNeighborsClassifier

# Here are tests for the KNeighborsClassifier implementation.

class Test_KNeighborsClassifier(unittest.TestCase):

    def test_KNeighborsClassifier_concept(self):
        '''Test concept vs manually derived results'''
        X_s1 = [[1]]
        y_s1 = [1]
        X_s2 = [[0], [1], [2], [3]]
        y_s2 = [0, 0, 1, 1]
        X_s3 = [[0, 0], [1, 0], [3, 4.5], [1, 1], [-2, -3]]
        y_s3 = ['a', 'a', 'b', 'b', 'c']
        y_s4 = [['a', 1], ['a', 0], ['b', 1], ['b', 0], ['c', 1]]

        # Concept for kneighbors 1 training point of 1 dimension
        nn_1 = KNeighborsClassifier(n_neighbors=1)
        nn_1.fit(X_s1,y_s1)

        # Test Distance Matrix
        self.assertTrue(np.equal(np.array([1]), nn_1.kneighbors([[2]]))[0])
        # Test Index Matrix
        self.assertTrue(np.equal(np.array([0]), nn_1.kneighbors([[2]]))[1])
        # Test Classification
        self.assertTrue(np.equal(np.array(1), nn_1.predict([[2]])))
        # Test predicted probabilities
        self.assertTrue(np.equal(np.array(1), nn_1.predict_proba([[2]])))

        # Concept for predict, predict_proba 4 training points of 1 dimension
        nn_2 = KNeighborsClassifier(n_neighbors=3)
        nn_2.fit(X_s2, y_s2)

        # Test Distance Matrix
        self.assertTrue(np.allclose(np.array([[0.1, 0.9, 1.1], [0.1, 0.9, 1.1]]),
                                    nn_2.kneighbors([[1.1], [2.1]], sort_results=True)[0]))
        # Test Index Matrix
        self.assertTrue(np.allclose(np.array([[1, 2, 0], [2, 3, 1]]), nn_2.kneighbors([[1.1], [2.1]],
                                                                            sort_results=True)[1]))
        # Test Classification
        self.assertTrue(np.allclose(np.array([0, 1]), nn_2.predict([[0.9], [2.1]])))
        # Test predicted probabilities
        self.assertTrue(np.allclose(np.array([[2/3, 1/3], [1/3, 2/3]]), nn_2.predict_proba([[0.9], [2.1]])))

        # Concept for kneighbors 5 training points of 2 dimension incl negative values and string labels
        nn_3 = KNeighborsClassifier(n_neighbors=3)
        nn_3.fit(X_s3, y_s3)
        # Test Distance Matrix
        self.assertTrue(np.allclose(np.array([[1, 1, 1.41421356],
                                              [1, 2.82842712, 3.60555128]]),
                                    nn_3.kneighbors([[0, 1], [-2, -2]], sort_results=True)[0]))
        # Test Index Matrix
        self.assertTrue(np.allclose(np.array([[0, 3, 1],
                                              [4, 0, 1]]),
                                    nn_3.kneighbors([[0, 1], [-2, -2]], sort_results=True)[1]))

        # Test Classification
        self.assertTrue(np.array_equal(np.array(['a', 'a']), nn_3.predict([[0, 1], [-2, -2]])))
        # Test predicted probabilities
        self.assertTrue(np.allclose(np.array([[2/3, 1/3, 0],
                                              [2/3, 0, 1/3]]), nn_3.predict_proba([[0, 1], [-2, -2]])))

        # Concept for kneighbors 5 training points of 2 dimension incl negative values and multiple classification
        nn_4 = KNeighborsClassifier(n_neighbors=3)
        nn_4.fit(X_s3, y_s4)

        # Test Distance Matrix
        self.assertTrue(np.allclose(np.array([[1, 1, 1.41421356],
                                           [1, 2.82842712, 3.60555128]]),
                                    nn_4.kneighbors([[0, 1], [-2, -2]], sort_results=True)[0]))
        # Test Index Matrix
        self.assertTrue(np.allclose(np.array([[0, 3, 1],
                                              [4, 0, 1]]),
                                    nn_4.kneighbors([[0, 1], [-2, -2]], sort_results=True)[1]))
        # Test Classification
        self.assertTrue(np.array_equal(np.array([['a', 0],
                                                 ['a', 1]]), nn_4.predict([[0, 1], [-2, -2]])))

        # Test predicted probabilities: Have to test the two separate classifications individually,
        # as the probabilities are returned as arrays in a list
        self.assertTrue(np.allclose(np.array([[2/3, 1/3, 0],
                                              [2/3, 0, 1/3]]), nn_4.predict_proba([[0, 1], [-2, -2]])[0]))
        self.assertTrue(np.allclose(np.array([[2/3, 1/3],
                                              [1/3, 2/3]]), nn_4.predict_proba([[0, 1], [-2, -2]])[1]))


class Test_KNeighborsClassifier_sklearn_simbsig(unittest.TestCase):
    def test_KNeighborsClassifier_sklearn_simbsig(self):
        '''Compare simbsig.neighbors.KNeighborsClassifier vs sklearn.neighbors.KNeighborsClassifier as gold standard'''

        ################################################################################################################
        # Create Dataset
        np.random.seed(98)
        n_samples = 100
        n_queries = 50
        n_dim = 10
        n_classes = 5
        n_classifications = 2 # 2 classifications for integer labels, 1 classification for character label are tested
        X_train = np.random.uniform(low=-5, high=5, size=(n_samples, n_dim))

        X_query = np.random.uniform(low=-5, high=5, size=(n_queries, n_dim))

        # Covariance Matrix and Inverted Covariance Matrix
        if n_dim > 1:
            VI = np.linalg.inv(np.cov(X_train.T))

        # feature_weights vector
        w = np.random.uniform(size=(n_dim))
        ###############################################################################################################
        # Create integer labelled y_train_integer:
        y_train_integer = np.random.randint(low=0, high=n_classes, size=(n_samples,1))
        y_train_integer_2 = np.random.randint(low=0, high=n_classes+1, size=(n_samples,1))
        y_train_integer = np.concatenate((y_train_integer,y_train_integer_2), axis=1)

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
        INF = 10000000 # sufficient as infinity for our tests
        for labels in ['integer', 'categorical']:
            if labels == 'integer':
                y_train = y_train_integer
            else:
                y_train = y_train_categorical

            for n_neighbors in [5, n_samples]:
                    # if n_dim == 1, omit mahalanobis distance (depends on covariance matrix) and omit
                    # cosine distance (is either 0 or 2 for points of 1 dimension)
                    if n_dim == 1:
                        metric_lys = ["euclidean", "minkowski", "manhattan"]
                    else:
                        metric_lys = ["mahalanobis","euclidean", "minkowski", "manhattan", "cosine"]
                    for metric in metric_lys:
                        for feature_weights in [None, w]:
                            for device in ['cpu']:#,'gpu']:
                                if labels == 'integer':
                                    mode_lys = ['arrays', 'hdf5']
                                else:
                                    mode_lys = ['arrays']

                                for mode in mode_lys:
                                    for batch_size in [30, INF]:
                                        for sample_weights in ['uniform', 'distance']:
                                            ################################################################################
                                            # Initialize settings for this innermost loop
                                            # set default p
                                            p = 1.5

                                            # information string for error message
                                            inf_string = f'n_neighbors: {n_neighbors} sample_weights: {sample_weights}' \
                                                         f'metric: {metric} ' \
                                                         f'(if minkowski: p={p})' \
                                                         f'feature_weights: {feature_weights is not None} ' \
                                                         f'device: {device} mode: {mode} batch_size: {batch_size} ' \
                                                         f'labels: {labels}' \

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
                                                y_train_used = y_train # np.array
                                                X_query_used = X_query # np.array
                                            elif mode == 'hdf5':
                                                X_train_used = train_data # h5py.File
                                                X_query_used = query_data # h5py.File

                                            ################################################################################
                                            # create NearestNeighbor object
                                            if metric == "mahalanobis":
                                                nn_sk = KNeighborsClassifier_sk(n_neighbors=n_neighbors,
                                                                                weights=sample_weights,
                                                                                metric=metric,
                                                                                metric_params={'VI': VI})
                                                nn_simbsig = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                                             weights=sample_weights,
                                                                             metric=metric,
                                                                             feature_weights=feature_weights,
                                                                             device=device, mode=mode,
                                                                             batch_size=batch_size,
                                                                             metric_params={'VI': VI},
                                                                             verbose=False)
                                            else:
                                                nn_sk = KNeighborsClassifier_sk(n_neighbors=n_neighbors,
                                                                                weights=sample_weights,
                                                                                metric=metric,
                                                                                p=p)
                                                nn_simbsig = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                                             weights=sample_weights,
                                                                             metric=metric,
                                                                             feature_weights=feature_weights,
                                                                             device=device,
                                                                             mode=mode,
                                                                             batch_size=batch_size, p=p,
                                                                             verbose=False)

                                            ################################################################################
                                            # fit KNeighborsClassifier objects
                                            nn_sk.fit(X_train_sklearn, y_train)
                                            if mode == 'arrays':
                                                nn_simbsig.fit(X_train, y_train)
                                            elif mode == 'hdf5':
                                                nn_simbsig.fit(X_train_used)

                                            ################################################################################
                                            # test predict
                                            y_pred_sk = nn_sk.predict(X_query_sklearn)
                                            y_pred_simbsig = nn_simbsig.predict(X_query_used)

                                            # Have tolerance for different prediction compared to sklearn due to
                                            # small, numerically caused distance differences
                                            different_labels = np.sum(y_pred_sk != y_pred_simbsig)
                                            same_labels = np.sum(y_pred_sk == y_pred_simbsig)

                                            # The assertTrue statement is executed if there is a perfect match (passed),
                                            # or more than 5% of all predicted labels do not match
                                            if different_labels == 0 or different_labels > n_queries/20:
                                                self.assertTrue(np.array_equal(y_pred_sk, y_pred_simbsig),
                                                                msg=inf_string)
                                            else:
                                                print(f'{inf_string} produced {different_labels}'
                                                      f'different labels (while {same_labels} labels match)')

                                            ################################################################################
                                            # test predict_proba
                                            y_pred_proba_sk = nn_sk.predict_proba(X_query_sklearn)
                                            y_pred_proba_simbsig = nn_simbsig.predict_proba(X_query_used)

                                            # Have tolerance for different probability predictions compared to sklearn due
                                            # to small, numerically caused distance differences
                                            proba_tolerance = 0.001

                                            # if number of classifications is 1 (the most common case)
                                            if n_classifications==1 and labels=='integer':
                                                different_probs = np.sum(abs(y_pred_proba_sk-y_pred_proba_simbsig) > proba_tolerance)
                                                same_probs = np.sum(abs(y_pred_proba_sk-y_pred_proba_simbsig) < proba_tolerance)

                                                # The assertTrue statement is executed if there is an allclose match (passed),
                                                # or more than 10% of all predicted probabilities do not match (fail)
                                                # (there are n_queries * n_classes many probabilities computed)
                                                if different_probs == 0 or different_probs > n_queries*n_classes*0.1:
                                                    self.assertTrue(np.allclose(y_pred_proba_sk, y_pred_proba_simbsig, rtol=1e-3),
                                                                    msg=inf_string)
                                                else:
                                                    print(f'{inf_string} produced {different_probs} probability labels with'
                                                          f'difference > {proba_tolerance}, (while {same_probs} probs match below {proba_tolerance}'
                                                          f'difference)')

                                            # else if multi-classification
                                            else:
                                                for i in range(n_classifications):
                                                    # for each classification, select the probabilities table from the
                                                    # generated probabilities tables list
                                                    y_pred_proba_sk_i = y_pred_proba_sk[i]
                                                    y_pred_proba_simbsig_i = y_pred_proba_simbsig[i]

                                                    different_probs = np.sum(
                                                        abs(y_pred_proba_sk_i - y_pred_proba_simbsig_i) > proba_tolerance)
                                                    same_probs = np.sum(
                                                        abs(y_pred_proba_sk_i - y_pred_proba_simbsig_i) < proba_tolerance)

                                                    # The assertTrue statement is executed if there is an allclose match (passed),
                                                    # or more than 10% of all predicted probabilities do not match (fail)
                                                    # (there are n_queries * n_classes many probabilities computed)
                                                    if different_probs == 0 or different_probs > n_queries * n_classes * 0.1:
                                                        self.assertTrue(
                                                            np.allclose(y_pred_proba_sk_i, y_pred_proba_simbsig_i, rtol=1e-3),
                                                            msg=inf_string)
                                                    else:
                                                        print(
                                                            f'{inf_string} produced {different_probs} probability labels with'
                                                            f'difference > {proba_tolerance}, (while {same_probs} probs match below {proba_tolerance} '
                                                            f'difference)')
        train_data.close()
        query_data.close()
if __name__ == '__main__':
    unittest.main()
