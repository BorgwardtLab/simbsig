# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

from simbsig.base._base import NeighborsBase
from simbsig.utils.utils import prepare_classification
import numpy as np

class NearestNeighbors(NeighborsBase):
    """Unsupervised learner performing neighbor searches. Implements batched data loading for big datasets and optional GPU accelerated computations

    Parameters

    :parameter n_neighbors: int, default=5
            Number of neighbors to search for during `kneighbors` queries.
    :parameter radius: float, default=1.0
            Dimension of the neighboring space in which to search for :meth:`radius_neighbors`
            queries.
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

    def __init__(self, n_neighbors=5, radius=None, metric="euclidean", p=2, metric_params=None,
                 feature_weights=None, device='cpu', mode='arrays', n_jobs=0, batch_size=None, verbose=True, **kwargs):

        super().__init__(n_neighbors=n_neighbors, radius=radius, metric=metric, p=p, metric_params=metric_params,
                         feature_weights=feature_weights, device=device, mode=mode, n_jobs=n_jobs, batch_size=batch_size, verbose=verbose, **kwargs)

    def fit(self, X, y=None):
        """Fit the nearest neighbors estimator from the training dataset.

        Parameters

        :parameter X: array-like or h5py file handle.
                Training Data of shape (n_samples, n_features) or \
                (n_samples, n_samples) if `metric='precomputed'`
        :parameter y: Ignored.
                Only present by convention

        Returns

        :return self: NearestNeighbor
                The fitted nearest neighbors estimator.
        """
        value = self._fit(X, y)

        return value


    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, sort_results=False):
        """Find the K-neighbors of a point, with `K=n_neighbors`.
        Returns indices (including or not corresponding distances) of the K-neighbors.

        Parameters

        :parameter X: array-like or h5py file handle, shape (n_queries, n_features), \
            or (n_queries, n_indexed) if `metric == 'precomputed'`, \
                default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned, excluding itself.
        :parameter n_neighbors: int, default=None
            Number of neighbors to search for. By default, the value passed
            to the constructor is used.
        :parameter return_distance: bool, default=True
            Should the distances between the point and its neighbors be returned or not.
        :parameter sort_results: bool, default=False
            Should the nearest neighbors be sorted by increasing distance to the query
            point or not. Note that if `return_distance=False`and `sort_results=True`,
            an error will be returned.

        Returns

        :return neigh_ind: ndarray of shape (n_queries, n_neighbors)
            storing indices of the nearest neighbors in the population matrix.
        :return neigh_dist:  ndarray of shape (n_queries, n_neighbors)
            If `return_distance=True`: array representing the distances to points.

        """

        # run kneighbors query from base class
        # Allow X_query=None only if metric==precomputed
        if X is None and self.metric != 'precomputed':
            raise ValueError(f"Calling kneighors without query matrix X as argument only allowed if metric=='precomputed'.")

        return self._kneighbors(X, n_neighbors, return_distance, sort_results)

    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        """Find the neighbors within a given radius of a point or points.
        Returns indices (including or not corresponding distances) of the neighbors
        lying in or on the boundary of a ball with size ``radius`` around the points
        of the query array. Note that the result points might *not* be sorted by distance
        to their query point.

        Parameters

        :parameter X: array-like or h5py file handle of (n_samples, n_features), default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned, excluding itself.
        :parameter radius: float, default=None
            Dimension of the neighboring space in which the search is performed.
            By default, the value passed to the constructor is used.
        :parameter return_distance: bool, default=True
            Should the distances between the point and its neighbors be returned or not.
        :parameter sort_results: bool, default=False
            Should the nearest neighbors be sorted by increasing distance to the query
            point or not. Note that if `return_distance=False`and `sort_results=True`,
            an error will be returned.

        Returns

        :return neigh_dist: ndarray of shape (n_samples,) representing the distances to points.
            Only present if `return_distance=True`.
        :return neigh_ind: ndarray of shape (n_samples,) of arrays of indices of the approximate
            nearest points that lie within or at the border of a ball of size ``radius``
            around the query points.

        Notes
        -----
        Results from different points may not collect the same number of neighbors
        and therefore may not fit in a standard array.
        To overcome this problem efficiently, `radius_neighbors` returns
        an array containing 1D arrays of indices or distances.
        """

        # run kneighbors query from base class
        return super()._radius_neighbors(X, radius=radius, return_distance=return_distance, sort_results=sort_results)
