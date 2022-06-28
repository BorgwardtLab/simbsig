# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

"""Nearest Neighbor Classification"""

from simbsig.neighbors.NearestNeighbors import NearestNeighbors
from simbsig.utils.datasets import arraysDataset, hdf5Dataset
from simbsig.utils.utils import prepare_classification

import numpy as np
import torch

class KNeighborsClassifier(NearestNeighbors):
    """Vote-based classifier among the k-nearest neighbors, with `k=n_neighbors`.

    Parameters

    :parameter n_neighbors: int, default=5
        Number of neighbors to search for during :meth: `kneighbors` queries.
    :parameter metric: str or callable, default='minkowski'
        The distance metric used to quantify similarity between objects,
        with default metric being minkowski. Other available metrics include
        [‘euclidean’, ‘manhattan’, ‘minkowski’,’fractional’,’cosine’,’mahalanobis’].
        When `metric='precomputed'`, provide X as a distance matrix which will
        be square during fit.
    :parameter p: int, default=2
        Parameter to be used when `metric=’minkowski’`. Note that if `p=1` or `p=2`,
        it is equivalent to using `metric=‘manhattan’` (L1) or `metric=‘euclidean’`
        (L2), respectively. For any other arbitrary p, minkowski distance (L_p) is used.
    :parameter metric_params: dict, default=None
        Additional metric-specific keyword arguments.
    :parameter feature_weights: np.array of floats, default=None
        Vector giving user-defined weights to every feature.
        Must be of similar length as the number of features n_features_in.
        If `feature_weights=None`, uniform weights are applied.
    :parameter sample_weights: str or callable, default='uniform'
        Options supported are: [‘uniform’,’distance’] or callable
        Defines the weights to be applied to the nearest neighbors identified in the training set.
        If `weights='uniform'`, all points in each neighborhood are weighted equally.
        If `weights='distance'`, weight is proportional to the distance to the query point,
        such that neighbors closer to the query point have a greater influence on the prediction.
        If `weight='callable'`, a user-defined function should be passed.
        It requires to take array of distances as inputs and to return an equal-size array of weights.
    :parameter device: str, default='cpu'
        Which device to use for distance computations.
        Options supported are: [‘cpu’,’gpu’]
    :parameter mode: str, default='arrays'
        Whether the input data is in memory (as lists, arrays or tensors) or
        on disk as hdf5 files. The latter should be favored for big datasets.
        Options supported are: [‘arrays’,’hdf5’]
    :parameter n_jobs: int, default=0
        Number of jobs active in torch.dataloader.
    :parameter batch_size: str, default=None
        Batch size of data chunks that are processed at once for distance computations. Should be
        optimized for dataset when using `device='gpu'`.
        If `batch_size=None`, the entire dataset is loaded and processed at once,
        which may return an error when using `device='gpu'`.
    :parameter verbose: bool, default=True
        Logging information. If True, progression updates are produced.
    """

    def __init__(self, n_neighbors=5, metric="euclidean", p=2, metric_params=None,
                 feature_weights=None, weights='uniform', device='cpu', mode='arrays', n_jobs=0, batch_size=None, verbose=True, **kwargs):

        super().__init__(n_neighbors=n_neighbors, metric=metric, p=p, metric_params=metric_params,
                         feature_weights=feature_weights, device=device, mode=mode, n_jobs=n_jobs,
                         batch_size=batch_size, verbose=verbose, **kwargs)

        if weights not in ['uniform', 'distance'] or callable(weights):
            raise ValueError("'weights' should be either 'uniform', 'distance', or a callable")

        self.sample_weights = weights

    def fit(self, X, y=None):
        """Fit classifier based on the k-nearest neighbors from the training dataset.

        Parameters

        :parameter X: Training data passed in an array-like or h5py file format.
            Should be of shape (n_samples, n_features) or (n_samples, n_samples) if `metric='precomputed'`.
        :parameter y: None if X is a h5py file handle, array-like otherwise.
            Should be of shape (n_samples,) or (n_samples, n_outputs)

        Returns

        :return self: NearestNeighbor
            The fitted nearest neighbors estimator.
        """
        # self as classifier
        self._set_as_classifier()

        value = self._fit(X)

        # choose appropriate dataset
        if self.mode == 'arrays' and y is None:
            raise ValueError('y not specified.')
        elif self.mode == 'hdf5':
            set_type = "train"
            kwargs = {'X_path':self.X_path,'y_path':self.y_path,'set_type':set_type}

            # unpack y from hdf5 dataset
            y = hdf5Dataset(X,kwargs=kwargs).y[:]

        self.classes, self.y = prepare_classification(y)

        return value

    def predict(self, X=None):
        """Predict the target for the query dataset.

        Parameters

        :parameter X: Test samples passed in an array-like format or as h5py file handle.
            Should be of shape (n_queries, n_features) or (n_queries, n_indexed) if `metric == 'precomputed'`.

        Returns

        :return y: Predicted class labels for each sample returned as an
            ndarray of shape (n_queries,) or (n_queries, n_outputs).
        """

        # Allow X_query=None only if metric==precomputed
        if X is None and self.metric != 'precomputed':
            raise ValueError(f"Calling predict without query matrix X as argument only allowed if metric=='precomputed'.")

        neigh_dist, neigh_ind = self._kneighbors(X)

        n_classifications = len(self.classes)

        set_type = "query"
        kwargs = {'X_path':self.X_path,'y_path':self.y_path,'set_type':set_type}
        n_samples = len(eval(f'{self.mode}Dataset(X,kwargs=kwargs)'))
        y_pred = np.empty((n_samples, n_classifications), dtype=self.classes[0].dtype)

        # if sample_weights are uniform, the mode is fast
        if self.sample_weights == 'uniform':

            # This loop iterates only once if single classification
            for k, classes_k in enumerate(self.classes):
                a = self.y[:, k][neigh_ind]
                b = np.array(torch.mode(torch.tensor(a), axis=1)[0])

                y_pred[:, k] = classes_k[b]

        # if sample_weights 'distance' or callable, use predict_proba method
        else: # sample weights distance or callable
            probs = self.predict_proba(X)
            # if len(self.classes) is 1, then have to put probs in a list for generalization
            if len(self.classes) == 1:
                probs = [probs]

            # if single classification, the k-loop iterates only once
            for k, classes_k in enumerate(self.classes):
                max_probs = np.argmax(probs[k], axis=1)
                y_pred[:,k] = classes_k[max_probs]

        if y_pred.size == y_pred.shape[0]:
            return y_pred.ravel()
        else:
            return y_pred

    def predict_proba(self, X=None):
        """Return probability estimates for each class for each sample from the testing datatset.

        Parameters

        :parameter X: Test samples passed in an array-like or h5py file format.
            Should be of shape (n_queries, n_features) or (n_queries, n_indexed) if `metric == 'precomputed'`.

        Returns

        :return p: Predicted class probabilities for each sample returned as an
            ndarray of shape (n_queries, n_classes) or a list of n_outputs
            of such arrays if n_outputs > 1.
            Note that classes are returned according to lexicographic order.
        """

        # Allow X_query=None only if metric==precomputed
        if X is None and self.metric != 'precomputed':
            raise ValueError(f"Calling predict_proba without query matrix X as argument only allowed if metric=='precomputed'.")

        # run kneighbors query from base class
        neigh_dist, neigh_ind = self._kneighbors(X)

        set_type='query'
        kwargs = {'X_path': self.X_path, 'y_path': self.y_path, 'set_type': set_type}
        n_queries = len(eval(f'{self.mode}Dataset(X,kwargs=kwargs)'))

        if self.sample_weights == 'uniform':
            weights = (1./self.n_neighbors)*np.ones_like(neigh_dist)
        elif self.sample_weights == 'distance':
            # Add a small number for numerical stability
            weights = np.divide(1./(neigh_dist+1e-8),np.sum(1./(neigh_dist+1e-8),axis=1,keepdims=True))
        elif callable(self.sample_weights):
            weights = self.sample_weights(neigh_dist)

        probs = []

        # if single classification, the k-loop iterates only once
        for k, cls_k in enumerate(self.classes):
            y_pred = self.y[:, k][neigh_ind]
            probs_k = np.zeros((n_queries, len(cls_k)))

            # itertates through neighbors
            for i, idx in enumerate(y_pred.T):  # loop is O(n_neighbors)
                # iterates through labels
                for row in range(len(y_pred)):
                    probs_k[row, idx[row]] += weights[row, i]

            # make "probabilities" from the labels of the nearest neighbors
            tot_sum = probs_k.sum(axis=1, keepdims=True)
            tot_sum[tot_sum == 0] = 1
            probs_k /= tot_sum

            probs.append(probs_k.squeeze())

        if len(probs) == 1:
            return probs[0]
        else:
            return probs
