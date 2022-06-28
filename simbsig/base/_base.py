"""
# Author: Eljas Roellin

"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from simbsig.utils.metrics import DistanceMetrics
from simbsig.utils.datasets import arraysDataset,hdf5Dataset

class NeighborsBase:
    """Private basis class which implements batched data loading for big datasets and optional GPU accelerated computations

    Parameters
    :parameter n_neighbors: int, default=5
            Number of neighbors to search for during :meth: `kneighbors` queries.
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

    """

    def __init__(self, n_neighbors=5, radius=None, metric="minkowski", p=2, metric_params=None,
                 feature_weights=None, device='cpu', mode='arrays', n_jobs=0, batch_size=None, verbose=True, **kwargs):

        self.n_neighbors = n_neighbors
        self.radius = radius
        self.metric = metric
        # Default metric: minkowski with p=2, which is equivalent to "euclidean"
        if metric_params is None:
            self.metric_params = {'p':p}
        else:
            self.metric_params = metric_params
            self.metric_params.update({'p':p})


        self.n_jobs = n_jobs # for dataloaders
        self.mode = mode
        self.batch_size = batch_size
        self.verbose = verbose
        self._estimator_type = None
        self.device = torch.device('cuda') if device == 'gpu' else torch.device('cpu')
        self.X_path = kwargs.pop('X_path', 'X')
        self.y_path = kwargs.pop('y_path', 'y')
        self._radius_mode = False

        if feature_weights is not None:
            self.metric_params.update({'feature_weights':torch.tensor(feature_weights,dtype=torch.float32,device=self.device)})

    def _fit(self, X, y=None):
        """Fit the nearest neighbors estimator from the training dataset.

        Parameters
        :parameter X: array-like or h5py file handle.
                Training Data of shape (n_samples, n_features) or \
                (n_samples, n_samples) if `metric='precomputed'`
        :parameter y: If classifier or regressor, None if X is a h5py file handle, array-like otherwise.
            Should be of shape (n_samples,) or (n_samples, n_outputs).
            If NearestNeighbor search, y=None.

        Returns
        :return self: NearestNeighbor
                The fitted nearest neighbors estimator.
        """
        # Lazy step, just set up the train set
        path_dict = {'X_path': self.X_path, 'y_path': self.y_path}
        self.X_train_set = eval(f'{self.mode}Dataset(X,**path_dict)')

        return self


    def _set_as_classifier(self):
        """ Tag the instance as 'classifier'
        """
        self._estimator_type = "classifier"

    def _set_as_regressor(self):
        """ Tag the instance as 'regressor''
        """
        self._estimator_type = "regressor"


    def _kneighbors(self, X_query, n_neighbors=None, return_distance=True, sort_results=False):
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

        # Disallow n_neighbors > number of training samples if run kneighbors (do allow this in radius mode!)
        if not self._radius_mode and self.n_neighbors > len(self.X_train_set):
            raise ValueError(f"simbsig does not allow n_neighbors ({self.n_neighbors}) > number of training samples ({len(self.X_train_set)})")
            return

        # If metric=='precomputed', the 'training set size' is the dimension of the distance matrix - 1
        if not self._radius_mode and self.metric=='precomputed' and self.n_neighbors > len(self.X_train_set)-1:
            raise ValueError(f"simbsig does not allow n_neighbors ({self.n_neighbors}) > number of training samples ({len(self.X_train_set)-1})")
            return

        # The metrics used are in DistanceMetrics
        metrics = DistanceMetrics(device=self.device)

        # X_query may be None, if metric=='precomputed'.
        if X_query is not None:
            path_dict = {'X_path': self.X_path, 'y_path': self.y_path}
            X_query_set = eval(f'{self.mode}Dataset(X_query,**path_dict)')

        # If n_neighbors is None, the n_neighbors from the constructor is used
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # If the batch_size attribute of the instance is None, process entire dataset at once.
        if self.batch_size is None:
            if X_query is not None:
                self.batch_size = max(len(self.X_train_set), len(X_query_set))
            else:
                self.batch_size = len(self.X_train_set)

        # If self.metric == precomputed, the distance matrix is expected to be the input already
        if self.metric == 'precomputed':
            return self._precomputed(X_query, n_neighbors, return_distance, sort_results)

        X_train_loader = DataLoader(self.X_train_set,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.n_jobs)

        X_query_loader = DataLoader(X_query_set,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.n_jobs)


        # If a custom metric function is provided, use this
        if callable(self.metric):
            self.dist_fun = self.metric
        # Choose appropriate metric
        else:
            self.dist_fun = eval(f'metrics._{self.metric}')

        # Initialize empty containers for the return value:
        # in kneighbor-mode: n_neighbour many coloumns
        # in radius-mode: array of dtype object
        if not self._radius_mode:
            neigh_dist_ = np.zeros((len(X_query_set), n_neighbors))
            neigh_ind_ = np.zeros((len(X_query_set), n_neighbors)).astype(int)

        elif self._radius_mode and self.radius >= 0.:
            # empty, because we fill this with lists/arrays of variable length
            neigh_dist_ = np.empty(len(X_query_set), dtype=object)
            neigh_ind_ = np.empty(len(X_query_set), dtype=object)
        else: # self._radius_mode and self.radius < 0
            raise ValueError('The radius needs to be >=0.')

        # Dont do argpartition if n_neighbors it the number of samples in the train set
        do_argpartition = True
        if self.n_neighbors == len(self.X_train_set):
            do_argpartition = False

        query_idx = 0
        # Process query data batch-wise
        for X_query_batch in tqdm(X_query_loader,desc='X_query progress',leave=False,disable=True if not self.verbose else False):
            start_idx = 0
            # Move query batch to device
            X_query_batch = X_query_batch.to(self.device)
            neigh_dist_line = np.zeros((len(X_query_batch), len(self.X_train_set)))

            # Process train data batch-wise
            for X_train_batch in tqdm(X_train_loader,desc='X_train for X_query progress', leave=False,disable=True if not self.verbose else False):
                # Move train batch to device
                X_train_batch = X_train_batch.to(self.device)
                neigh_dist_line[:, start_idx:start_idx + len(X_train_batch)] = self.dist_fun(X_query_batch,
                                                                                             X_train_batch,
                                                                                             **(self.metric_params))
                start_idx += X_train_batch.shape[0]

            # if kneighbors-style query
            if not self._radius_mode:
                # For each point from query batch: find the 'n_neighbors' nearest neighbors.
                # Each row i in neighbor_idxs corresponds to the 'n_neighbors' nearest neighbors of query point i.
                # For example, the first entry of row i is the index of once nearest neighbor of query point i in
                # the train set.

                # if sort_results: do sorting. else: do the (faster) argpartition.
                # note the result for classification and regression is equivalent, up to potentially different
                # choices among neighbors of equal distance.
                if self.device == torch.device('cuda'):
                    neigh_dist_line_tensor = torch.tensor(neigh_dist_line).to(self.device)
                    if sort_results:
                        neighbor_idxs = torch.topk(neigh_dist_line_tensor, n_neighbors, largest=False,
                                                   sorted=True, axis=1)[1].cpu().numpy()
                    else:
                        if do_argpartition:
                            neighbor_idxs = torch.topk(neigh_dist_line_tensor, n_neighbors, largest=False,
                                                       sorted=False, axis=1)[1].cpu().numpy()
                        else:
                            neighbor_idxs = np.meshgrid(np.arange(0, n_neighbors), np.zeros(len(neigh_dist_line)))[0]

                else: # if self.device==torch.device('cpu')
                    if sort_results:
                        #neighbor_idxs = torch.topk(torch.from_numpy(neigh_dist_line), n_neighbors, largest=False, sorted=True, axis=1)[1].cpu().numpy()
                        neighbor_idxs = np.argsort(neigh_dist_line, axis=-1)[:, :n_neighbors]
                    else:
                        if do_argpartition:
                            #neighbor_idxs = torch.topk(torch.from_numpy(neigh_dist_line), n_neighbors, largest=False, sorted=False, axis=1)[1].cpu().numpy()
                            neighbor_idxs = np.argpartition(neigh_dist_line,n_neighbors, axis=-1)[:, :n_neighbors]
                        else:
                            neighbor_idxs = np.meshgrid(np.arange(0,n_neighbors), np.zeros(len(neigh_dist_line)))[0]

                # for each query point (row in neigh_dist_line) select its n_neighbors many
                # neighbors (coloumns in neigh_dist_line)
                neigh_dist_line = np.take_along_axis(neigh_dist_line, neighbor_idxs, axis=1)

                # Iteratively fill the final m x k neigh_dist_ matrix and its corresponding m x k neigh_ind_ matrix
                neigh_dist_[query_idx:query_idx + X_query_batch.shape[0], :] = neigh_dist_line
                neigh_ind_[query_idx:query_idx + X_query_batch.shape[0], :] = neighbor_idxs

            # if radius_neighbors-style query
            else:
                # If metric=='precomputed' for each query point, screen its neighbor's distances for being within radius
                for i in range(neigh_dist_line.shape[0]):
                    idxs = (neigh_dist_line[i, :] <= self.radius)

                    if sort_results:
                        relevant_neigh_dist = neigh_dist_line[i,:][idxs]
                        relevant_neigh_ind = np.where(idxs == 1)[0]

                        idxs_sorting_distance = np.argsort(relevant_neigh_dist)

                        neigh_dist_[query_idx + i] = relevant_neigh_dist[idxs_sorting_distance]
                        neigh_ind_[query_idx + i] = relevant_neigh_ind[idxs_sorting_distance]
                    else:
                        neigh_dist_[query_idx+i] = neigh_dist_line[i, :][idxs]
                        neigh_ind_[query_idx+i] = np.where(idxs==1)[0]

            query_idx += X_query_batch.shape[0]

        self.neigh_dist_ = neigh_dist_
        self.neigh_ind_ = neigh_ind_

        # Set back _radius_mode to False, in case a kneighbors query follows on this fitted object
        self._radius_mode = False

        if return_distance:
            return neigh_dist_, neigh_ind_
        else:
            return neigh_ind_

    def _precomputed(self, X_query, n_neighbors, return_distance, sort_results):
        """Base class for precomputed functionality
        """
        # if X_query is None, the training data's nearest neighbors should be returned
        if X_query is None:
            X_query_loader = DataLoader(self.X_train_set,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.n_jobs)
            n_samples = len(self.X_train_set)

        else:
            path_dict = {'X_path': self.X_path, 'y_path': self.y_path}
            X_query_set = eval(f'{self.mode}Dataset(X_query,**path_dict)')

            X_query_loader = DataLoader(X_query_set,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.n_jobs)
            n_samples = len(X_query_set)

        # Initialize empty containers for the return value:
        # in kneighbor-mode: n_neighbour many coloumns
        # in radius-mode: array of dtype object
        if not self._radius_mode:
            neigh_dist_ = np.zeros((n_samples, n_neighbors))
            neigh_ind_ = np.zeros((n_samples, n_neighbors)).astype(int)

        elif self._radius_mode and self.radius >= 0.0:
            # empty, because we fill this with lists/arrays of variable length
            neigh_dist_ = np.empty(n_samples, dtype=object)
            neigh_ind_ = np.empty(n_samples, dtype=object)

        else: # self._radius_mode and self.radius < 0.0
            raise ValueError('The radius needs to be >= 0, but is {self.radius}.')

        # Dont do argpartition if n_neighbors it the number of samples in the train set
        # Notice that when X is None, the one of the n_sample points is the query point itself
        do_argpartition = True
        if X_query is None and self.n_neighbors == n_samples-1:
            do_argpartition = False
        elif self.n_neighbors == n_samples:
            do_argpartition = False

        query_idx = 0
        # Process query data batch-wise
        for X_query_batch in tqdm(X_query_loader, desc='X_query progress', leave=False, disable=True if not self.verbose else False):
            X_query_batch = X_query_batch.numpy()

            # If X_query is None, ignore the diagonal elements in the square training data matrix by setting them
            # to float('inf), as these are the points for which the neighbors are queried and should be ignored
            # as query result.
            # Below, this opertation is done accounting for  the diagonal matrix's rows being loaded
            # batchwise.
            if X_query is None:
                batch_len = X_query_batch.shape[0]
                left_submatrix = X_query_batch[:, :query_idx]
                # get submatrix
                middle_submatrix = X_query_batch[:, query_idx:query_idx + batch_len]
                # delete its diagonal
                middle_submatrix[np.eye(batch_len, dtype=bool)] = float('inf')
                # reshape the resulting 1D vector again into corrected matrix format
                middle_submatrix = middle_submatrix.reshape(batch_len, -1)
                right_submatrix = X_query_batch[:, query_idx + batch_len:]
                # concatenate these again
                X_query_batch = np.concatenate((left_submatrix, middle_submatrix, right_submatrix), axis=1)

            # if kneighbors-style query
            if not self._radius_mode:
                # For each point from query batch: find the 'n_neighbors' nearest neighbors.
                # Each row i in neighbor_idxs corresponds to the 'n_neighbors' nearest neighbors of query point i.
                # For example, the first entry of row i is the index of once nearest neighbor of query point i in
                # the train set.
                if sort_results:
                    neighbor_idxs = np.argsort(X_query_batch, axis=1)[:, :n_neighbors]

                else:
                    if do_argpartition:
                        neighbor_idxs = np.argpartition(X_query_batch, n_neighbors, axis=1)[:, :n_neighbors]
                    else:
                        # Here indices for the entire batch are re-created, only deleting the indices corresponding
                        # to the diagonals in the squared training data matrix
                        # Below, this opertation is done accounting for  the diagonal matrix's rows being loaded
                        # batchwise.
                        neighbor_idxs = np.meshgrid(np.arange(0,n_neighbors+1), np.zeros(len(X_query_batch)))[0]
                        batch_len = neighbor_idxs.shape[0]
                        left_submatrix = neighbor_idxs[:, :query_idx]
                        # get submatrix
                        middle_submatrix = neighbor_idxs[:, query_idx:query_idx + batch_len]
                        # delete its diagonal
                        middle_submatrix = middle_submatrix[~np.eye(batch_len, dtype=bool)]
                        # reshape the resulting 1D vector again into corrected matrix format
                        middle_submatrix = middle_submatrix.reshape(batch_len, -1)
                        right_submatrix = neighbor_idxs[:, query_idx + batch_len:]
                        # concatenate these again
                        neighbor_idxs = np.concatenate((left_submatrix, middle_submatrix, right_submatrix), axis=1)

                # for each query point (row in neigh_dist_line) select its n_neighbors many
                # neighbors (coloumns in neigh_dist_line)
                if n_neighbors==1:
                    neighbor_idxs = np.reshape(neighbor_idxs, (-1,1))
                neigh_dist_line = np.take_along_axis(X_query_batch, neighbor_idxs, axis=1)
                # neigh_dist_line = np.array([X_query_batch[i, idxs] for i, idxs in enumerate(neighbor_idxs)])

                neigh_dist_[query_idx:query_idx + X_query_batch.shape[0]] = neigh_dist_line
                neigh_ind_[query_idx:query_idx + X_query_batch.shape[0]] = neighbor_idxs

            # if radius_neighbors-style query
            else:
                for i in range(X_query_batch.shape[0]):
                    idxs = (X_query_batch[i, :] <= self.radius)
                    neigh_dist_[query_idx+i] = X_query_batch[i, :][idxs]
                    neigh_ind_[query_idx+i] = np.where(idxs==1)[0]

            query_idx += X_query_batch.shape[0]

        self.neigh_ind_ = neigh_ind_
        self.neigh_dist_ = neigh_dist_

        if return_distance:
            return neigh_dist_, neigh_ind_
        else:
            return neigh_ind_

    def _radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
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

        # Overwrite radius of object permanently
        if radius is not None:
            self.radius = radius

        if self.radius is None:
            raise ValueError(f'The radius needs to be >= 0, but is {self.radius}')

        # Tag this mode as True
        self._radius_mode = True

        return self._kneighbors(X, n_neighbors=None, return_distance=return_distance, sort_results=sort_results)
