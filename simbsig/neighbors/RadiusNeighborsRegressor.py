# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

"""Nearest Neighbor Regression. """

from simbsig.base._base import NeighborsBase
from simbsig.utils.datasets import arraysDataset,hdf5Dataset
import numpy as np
import warnings

class RadiusNeighborsRegressor(NeighborsBase):
    """Regression based on neighbors located in a fixed, user-defined neighboring space.
     The target is predicted based on the nearest neighbors' target identified in the training set.

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

    def __init__(self, radius=1.0, *, weights='uniform', p=2, metric='euclidean', metric_params=None,
                 feature_weights=None, device='cpu', mode='arrays', n_jobs=0, batch_size=None, verbose=True, **kwargs):

        super().__init__(radius=radius, metric=metric, p=p, metric_params=metric_params,
                         feature_weights=feature_weights, device=device, mode=mode, n_jobs=n_jobs, batch_size=batch_size,
                         verbose=verbose, **kwargs)
        if weights not in ['uniform', 'distance'] or callable(weights):
            raise ValueError("'weights' should be either 'uniform', 'distance', or a callable")

        self.sample_weights = weights

    def fit(self, X, y=None):
        """Fit regressor based on the radius neighbors from the training dataset.

        Parameters

        :parameter X: Training data passed in an array-like or h5py file format.
            Should be of shape (n_samples, n_features) or (n_samples, n_samples) if `metric='precomputed'`.
        :parameter y: Target values from the training data passed in an array-like or sparse matrix format.
            Should be of shape (n_samples,) or (n_samples, n_regressions)

        Returns

        :return self: RadiusNeighborsRegressor
            The fitted radius neighbors regressor.
        """

        # self as classifier
        self._set_as_regressor()
        value = self._fit(X, y)

        if self.mode == 'arrays' and y is None:
            raise ValueError('y not specified.')
        elif self.mode == 'hdf5':
            set_type='train'
            kwargs = {'X_path':self.X_path,'y_path':self.y_path,'set_type':set_type}
            y = hdf5Dataset(X,kwargs=kwargs).y[:]

        # Convert y to numpy array as compatible container for subsequent operations
        y = np.array(y)

        # Guide for matching prediction dimension with input dimension
        self._y_ndim_1 = False
        if y.ndim == 1:
            self._y_ndim_1 = True
            self.y = y.reshape(-1,1)
        else:
            self.y = y

        return value

    def predict(self, X=None):
        """Predict the target for the testing dataset.

        Parameters

        :parameter X: Test samples passed in an array-like or h5py file format.
            Should be of shape (n_queries, n_features), or (n_queries, n_indexed) if `metric == 'precomputed'`.

        Returns

        :return y: Predicted target values of `dtype=double` returned as an
            ndarray of shape (n_queries,) or (n_queries, n_regressions).
        """

        # Allow X_query=None only if metric==precomputed
        if X is None and self.metric != 'precomputed':
            raise ValueError(f"Calling predict without query matrix X as argument only allowed if metric=='precomputed'.")

        neigh_dist, neigh_ind = self._radius_neighbors(X)

        if self.sample_weights == 'uniform':
            weights = np.array([(1./max(len(neigh_dist[i]), 1))*np.ones_like(neigh_dist[i])
                                for i in range(neigh_dist.shape[0])], dtype=object)
        elif self.sample_weights == 'distance':
            # Add a small number for numerical stability
            weights = np.array([np.divide(1./(neigh_dist[i]+1e-8), np.sum(1./(neigh_dist[i]+1e-8),
                                                                          axis=0, keepdims=True))
                                for i in range(neigh_dist.shape[0])], dtype=object)
        elif callable(self.sample_weights):
            weights = np.array([self.sample_weights(neigh_dist[i]) for i in range(neigh_dist.shape[0])],dtype=object)

        empty_obs = np.full_like(self.y[0], np.nan)

        y_pred = np.zeros((len(neigh_dist),len(self.y[0])))
        outliers = np.full(len(y_pred), False, dtype=bool)

        # If single regression, which is the most common use case, this outer loop iterates only once
        for j in range(self.y.shape[1]):
            # for each query point, evaluate its neighbors
            for i, ind in enumerate(neigh_ind):
                # if a query point has neighbors, do regression
                if len(ind) != 0:
                    y_pred[i,j] = np.sum(weights[i] * self.y[ind, j].ravel(), axis=0)
                # else if query point has no neighbors, assign nan value to prediction instead of regression value
                else:
                    outliers[i] = True
                    y_pred[i,:] = empty_obs

        y_pred = np.array(y_pred).astype(float)

        # Warn user which query points do not have any neighbors
        if not np.array_equal(outliers, np.full_like(outliers, False)):
            empty_query_point_neighborhood_msg = (
                f"The query points {np.arange(len(y_pred))[outliers]} have no neighbors"
                f" within radius {self.radius}, leading to no meaningful prediction at these query points."
            )
            warnings.warn(empty_query_point_neighborhood_msg)

        # If y during fit was 1-Dimensional (only possible for single regression), the y output of the prediction also
        # is 1-Dimensional
        if self._y_ndim_1:
            return y_pred.ravel()
        else:
            return y_pred
