# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

from simbsig.base._base import NeighborsBase
from simbsig.utils.datasets import arraysDataset,hdf5Dataset
from simbsig.utils.utils import prepare_classification
import torch

import numpy as np

class RadiusNeighborsClassifier(NeighborsBase):
    """Vote-based classifier among neighbors located in a user-defined radius.

    Parameters

    :parameter radius: float, default=1.0
        Dimension of the neighboring space in which to search for :meth:`radius_neighbors`
        queries.
    :parameter sample_weights: str or callable, default='uniform'
        Options supported are: [‘uniform’,’distance’] or callable
        Defines the weights to be applied to the nearest neighbors identified in the training set.
        If `weights='uniform'`, all points in each neighborhood are weighted equally.
        If `weights='distance'`, weight is proportional to the distance to the query point,
        such that neighbors closer to the query point have a greater influence on the prediction.
        If `weight='callable'`, a user-defined function should be passed.
        It requires to take array of distances as inputs and to return an equal-size array of weights.
    :parameter p: int, default=2
        Parameter to be used when `metric=’minkowski’`. Note that if `p=1` or `p=2`,
        it is equivalent to using `metric=‘manhattan’` (L1) or `metric=‘euclidean’`
        (L2), respectively. For any other arbitrary p, minkowski distance (L_p) is used.
    :parameter metric: str or callable, default='minkowski'
        The distance metric used to quantify similarity between objects,
        with default metric being minkowski. Other available metrics include
        [‘euclidean’, ‘manhattan’, ‘minkowski’,’fractional’,’cosine’,’mahalanobis’].
        When `metric='precomputed'`, provide X as a distance matrix which will
        be square during fit.
    :parameter metric_params: dict, default=None
        Additional metric-specific keyword arguments.
    :parameter feature_weights: np.array of floats, default=None
        Vector giving user-defined weights to every feature.
        Must be of similar length as the number of features n_features_in.
        If `feature_weights=None`, uniform weights are applied.
    :parameter device: str, default='cpu'
        Which device to use for distance computations.
        Options supported are: [‘cpu’,’gpu’]
    :parameter mode: str, default='arrays'
        Whether the input data is in memory (as lists, arrays or tensors) or
        on disk as hdf5 files. The latter should be favored for big datasets.
        Options supported are: [‘arrays’,’hdf5’]
    :parameter n_jobs: int, default=None
        Number of jobs active in torch.dataloader.
    :parameter batch_size: str, default=None
        Batch size of data chunks that are processed at once for distance computations. Should be
        optimized for dataset when using `device='gpu'`.
        If `batch_size=None`, the entire dataset is loaded and processed at once,
        which may return an error when using `device='gpu'`.
    :parameter verbose: bool, default=True
        Logging information. If True, progression updates are produced.

    """

    def __init__(self, radius=1.0, *, weights='uniform', outlier_label=None, metric='euclidean', p=2, metric_params=None,
                 feature_weights=None, device='cpu', mode='arrays', n_jobs=0, batch_size=None, verbose=True, **kwargs):

        super().__init__(radius=radius, metric=metric, p=p, metric_params=metric_params,
                         feature_weights=feature_weights, device=device, mode=mode, n_jobs=n_jobs,
                         batch_size=batch_size, verbose=verbose, **kwargs)
        if weights not in ['uniform', 'distance'] or callable(weights):
            raise ValueError("'weights' should be either 'uniform', 'distance', or a callable")

        self.sample_weights = weights
        self.outlier_label = outlier_label

    def fit(self, X, y=None):
        """Fit classifier based on the radius neighbors from the training dataset.

        Parameters

        :parameter X: Training data passed in an array-like or h5py file format.
            Should be of shape (n_samples, n_features) or (n_samples, n_samples) if `metric='precomputed'`.
        :parameter y: Target values from the training data passed in an array-like or sparse matrix format.
            Should be of shape (n_samples,) or (n_samples, n_outputs)

        Returns

        :return self: RadiusNeighborsClassifier
            The fitted radius neighbors classifier.
        """
        # self as classifier
        self._set_as_classifier()
        value = self._fit(X, y)

        if self.mode == 'arrays' and y is None:
            raise ValueError('y not specified.')
        elif self.mode == 'hdf5':
            set_type='train'
            kwargs = {'X_path':self.X_path,'y_path':self.y_path,'set_type':set_type}
            y = hdf5Dataset(X,kwargs=kwargs).y[:]
        self.classes, self.y = prepare_classification(y)

        return value

    def predict(self, X=None):
        """Predict the class labels for the testing dataset.

        Parameters

        :parameter X: Test samples passed in an array-like or h5py file format.
            Should be of shape (n_queries, n_features) or (n_queries, n_indexed) if `metric == 'precomputed'`.

        Returns

        :return y: Predicted class labels for each sample returned as an
            ndarray of shape (n_queries,) or (n_queries, n_outputs).
        """
        # Allow X_query=None only if metric==precomputed
        if X is None and self.metric != 'precomputed':
            raise ValueError(f"Calling predict without query matrix X as argument only allowed if metric=='precomputed'.")

        # run radius_neighbors
        neigh_dist, neigh_ind = self._radius_neighbors(X)

        n_outputs = len(self.classes)

        set_type = "query"
        kwargs = {'X_path':self.X_path,'y_path':self.y_path,'set_type':set_type}
        n_samples = len(eval(f'{self.mode}Dataset(X,kwargs=kwargs)'))

        y_pred = np.empty((n_samples, n_outputs), dtype=self.classes[0].dtype)

        if self.sample_weights == 'uniform':
            # This outer loop iterates only once if single classification
            for k, classes_k in enumerate(self.classes):
                outliers = np.full_like(y_pred[:, k], False, dtype=bool)
                for i, query_neigh in enumerate(neigh_ind):
                        if len(query_neigh) != 0:
                            a = int(torch.mode(torch.tensor(self.y[query_neigh, :]), axis=0)[0][k])
                            label = int((torch.mode(torch.tensor(self.y[query_neigh, :]), axis=0)[0][k]))
                            y_pred[i, k] = classes_k[label]
                        else:
                            outliers[i] = True
                            #y_pred[i, k] = self.outlier_label

                # Alert user on outlier points and point out options offered
                if not np.array_equal(outliers, np.full_like(outliers, False)) \
                        and self.outlier_label is None:
                    raise ValueError(f'Query points {np.arange(n_samples)[outliers]} do not'
                                     f' have any neighbors using radius {self.radius}: Use larger'
                                     f' radius or specify the argument "outlier_label".')
                # If an outlier label has been supplied and is necessary, use it
                elif not np.array_equal(outliers, np.full_like(outliers, False)):
                    y_pred[outliers,k] = self.outlier_label

        else: # if sample_weights == 'distance:
            # slower than the computations for uniform sample weights
            # take class with highest probability
            probs = self.predict_proba(X)
            if n_outputs == 1:
                probs = [probs]

            for i, probs_i in enumerate(probs):
                max_prob_idxs= np.argmax(probs_i, axis=1)
                y_pred[:,i] = self.classes[i][max_prob_idxs]
                outlier_query_idxs = np.isclose(np.max(probs_i, axis=1), 0)

                # Alert user on outlier points and point out options offered
                if not np.array_equal(outlier_query_idxs, np.full_like(outlier_query_idxs, False))\
                        and self.outlier_label is None:
                    raise ValueError(f'Query points {np.arange(0,n_samples)[outlier_query_idxs]} do not'
                                     f'have any neighbors using radius {self.radius}: Use larger'
                                     f'radius or specify the argument "outlier_label"')

                # If an outlier label has been supplied and is necessary, use it
                elif not np.array_equal(outlier_query_idxs, np.full_like(outlier_query_idxs, False)):
                    y_pred[outlier_query_idxs,i] = self.outlier_label

        if y_pred.size == y_pred.shape[0]:
            return y_pred.ravel()
        else:
            return y_pred

    # This method is taken almost without change from sklearn.
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

        # run radius_neighbors
        neigh_dist, neigh_ind = self._radius_neighbors(X)

        n_classifications = len(self.classes)

        set_type = "query"
        kwargs = {'X_path':self.X_path,'y_path':self.y_path,'set_type':set_type}
        n_queries = len(eval(f'{self.mode}Dataset(X,kwargs=kwargs)'))

        if self.sample_weights == 'uniform':
            weights = np.array([(1./max(len(neigh_dist[i]),1))*np.ones_like(neigh_dist[i]) for i in range(neigh_dist.shape[0])],dtype=object)
        elif self.sample_weights == 'distance':
            # Add a small number for numerical stability
            weights = np.array([np.divide(1./(neigh_dist[i]+1e-8),np.sum(1./(neigh_dist[i]+1e-8),axis=0,keepdims=True)) for i in range(neigh_dist.shape[0])],dtype=object)
        elif callable(self.sample_weights):
            weights = np.array([self.sample_weights(neigh_dist[i]) for i in range(neigh_dist.shape[0])],dtype=object)

        probs = []

        # if single classification, the k-loop iterates only once
        for k, cls_k in enumerate(self.classes):
            probs_k = np.zeros((n_queries, len(cls_k)))
            # O(number of query points)
            for i, query_neigh in enumerate(neigh_ind):
                pred_labels = self.y[query_neigh].astype(int)
                # iterates through labels
                for j, pred_label in enumerate(pred_labels):
                    probs_k[i, pred_label[k]] += weights[i][j]

            # normalize 'votes' into real [0,1] probabilities
            tot_sum = probs_k.sum(axis=1, keepdims=True)
            tot_sum[tot_sum == 0] = 1
            probs_k /= tot_sum

            probs.append(probs_k.squeeze())

        if len(probs) == 1:
            return probs[0]
        else:
            return probs
