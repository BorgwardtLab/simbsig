# Author: Michael Adamer
#         Eljas Roellin
#         Lucie Bourguignon
#
# License: BSD 3 clause

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from simbsig.utils.datasets import arraysDataset,hdf5Dataset
from simbsig.utils.metrics import DistanceMetrics

class MiniBatchKMeans:
    """KMeans class, implementing MiniBatchKMeans as described by Scully [1], batched data loading for big datasets and
    optional GPU accelerated computations.

    Parameters

    :parameter n_clusters: int, default=5
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
    :parameter max_iter: int, default=100
            Maximum number of iterations of the KMeans algorithm. Algorithm might terminate earlier, if tol is
            satisfied.
    :parameter tol: float, default=1e-5.
            Tolerance upon which KMeans stops iterating. If tolerance is not reached after max_iter many iterations,
            the algorithm terminates.
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
    :parameter random_state: int, default=None
            The random state for the seed of torch.
    :parameter init: obj, default='random'
            If 'random', cluster centers are selected uniformly at random from the training set.
            Alternatively, an array-like X of shape (n_clusters, n_features) can be passed which will
            be used as cluster initialization
    :parameter verbose: bool, default=True
            Logging information. If True, progression updates are produced.

    [1] Sculley, David. "Web-scale k-means clustering." Proceedings of the 19th international conference on
    World wide web. 2010.

    """
    def __init__(self, n_clusters=5, metric='euclidean', metric_params=None,
                 feature_weights=None, max_iter=100, tol=1e-2, device='cpu', mode='arrays',
                 n_jobs=0, batch_size=None, random_state=None, init='random', alpha = 0.95, verbose=True, **kwargs):

        self.n_clusters_ = n_clusters
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs # For dataloaders!
        self.max_iter = max_iter
        self.tol = 0.0 if tol is None else tol # Tolerance for when to stop iterating
        self.mode = mode
        self.batch_size = batch_size
        self.init = init
        self.device = torch.device('cuda') if device == 'gpu' else torch.device('cpu')
        self.alpha = alpha # weighting for ewma of delta
        self.verbose = verbose
        self.X_path = kwargs.pop('X_path', 'X')
        self.y_path = kwargs.pop('y_path', 'y')

        if feature_weights is not None:
            self.metric_params.update({'w':feature_weights})

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state) # seed for batching

    def fit(self,X,y=None):
        """Performs the MiniBatchKMeans algorithm with settings passed during init.

        Parameters

        :parameter X: array-like or h5py file handle.
                Training Data of shape (n_samples, n_features) or \
                (n_samples, n_samples) if `metric='precomputed'`
        :parameter y: Ignored.
                Only present by convention.

        Returns

        :return self: MiniBatchKMeans
                The MiniBatchKMeans object with computed cluster centers.
        """

        # Do not accept zero clusters
        if self.n_clusters_ == 0:
            raise ValueError(f'Number of clusters is {self.n_clusters_}, but should be >= 1')

        metrics = DistanceMetrics(device=self.device)

        path_dict = {'X_path':self.X_path,'y_path':self.y_path}
        X_train_set = eval(f'{self.mode}Dataset(X,**path_dict)')

        # Do not accept more clusters than training samples
        if self.n_clusters_ > len(X_train_set):
            raise ValueError(f'Number of clusters ({self.n_clusters_}) should be <= number of'
                             f' training samples ({len(X_train_set)})')

        # If the batch_size attribute of the instance is None, process entire dataset at once.
        if self.batch_size is None:
            self.batch_size = len(X_train_set)
        else:
            self.batch_size = min(self.batch_size,len(X_train_set))

        X_train_loader = DataLoader(X_train_set,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.n_jobs)

        # Choose appropriate metric
        if self.metric in 'precomputed':
            raise ValueError('This metric is not implemented.')
        elif callable(self.metric):
            self.dist_fun = self.metric
        else:
            self.dist_fun = eval(f'metrics._{self.metric}')

        if isinstance(self.init,str):
            if self.init == 'random':
                # If a random state is assigned during instantiation, the random initialization is seeded, too.
                # The RandomState Seed is taken to be the training set size

                random_sample = np.random.choice(range(len(X_train_set)), replace=False, size=self.n_clusters_)
                # random_sample indices need to be in ascending order
                random_sample = np.sort(random_sample)

                self.cluster_centers_ = torch.FloatTensor(X_train_set[random_sample]).to(self.device)

        elif isinstance(self.init,np.ndarray):
            self.cluster_centers_ = torch.FloatTensor(self.init).to(self.device)
        elif isinstance(self.init,torch.Tensor):
            self.cluster_centers_ = self.init.to(self.device)
        cluster_centers_new = self.cluster_centers_.clone()
        cluster_size_counter = torch.zeros(self.cluster_centers_.shape[0])

        step_counter = 0
        continue_flag = True

        # set up delta for ewma calculation
        delta = None

        while continue_flag:
            # Load data batch
            for i, data in tqdm(enumerate(X_train_loader),
                                leave=False,
                                desc=f'Epoch {int(step_counter/len(X_train_loader))}; Global step {step_counter}',
                                disable=True if not self.verbose else False):
                # Move data batch to device
                data = data.to(self.device)

                # Compute the cluster label
                dist_mat = self.dist_fun(data,self.cluster_centers_,**(self.metric_params if self.metric_params is not None else {}))
                assigned_clusters = np.argmin(dist_mat,axis=1)

                # Update the cluster centers
                for j,d in enumerate(data):
                    c = assigned_clusters[j]
                    cluster_size_counter[c] += 1
                    lr = 1./cluster_size_counter[c]
                    cluster_centers_new[c] = (1-lr)*cluster_centers_new[c] + lr*d

                # Compute the average change in centers
                if self.metric in ['minkowksi','euclidean','manhattan','hamming']:
                    normalizer = self._pairwise(cluster_centers_new,torch.zeros_like(cluster_centers_new,device=self.device),\
                                                **(self.metric_params if self.metric_params is not None else {}))
                    normalizer = normalizer.copy() # Make writeable
                    normalizer[normalizer == 0] = 1 # prevent division by zero
                else:
                    normalizer = torch.ones(self.n_clusters_,device=self.device)

                if delta is None:
                    delta = (self._pairwise(self.cluster_centers_,
                                            cluster_centers_new,
                                            **(self.metric_params if self.metric_params is not None else {}))/normalizer).max()

                else:
                    delta = self.alpha*(self._pairwise(self.cluster_centers_,
                                                       cluster_centers_new,
                                                       **(self.metric_params if self.metric_params is not None else {}))/normalizer).max() + (1-self.alpha)*delta

                self.cluster_centers_ = cluster_centers_new.clone()
                step_counter += 1

                # Stopping heuristic
                if delta <= self.tol:
                    continue_flag = False
                    break
                if int(step_counter/len(X_train_loader)) >= self.max_iter:
                    continue_flag = False
                    break

        self.cluster_centers_ = self.cluster_centers_.cpu().numpy()

    def predict(self,X):
        """Predicts for data of same dimension as training data to which cluster center its points belong.

        Parameters

        :parameter X: array-like or h5py file handle.
                Test Data of shape (n_samples, n_features)

        Returns

        :return clusters: array of integers
                The cluster centers of shape (n_samples,)
        """
        metrics = DistanceMetrics(device=self.device)

        path_dict = {'X_path':self.X_path,'y_path':self.y_path}
        X_query_set = eval(f'{self.mode}Dataset(X,**path_dict)')
        X_query_loader = DataLoader(X_query_set,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.n_jobs)

        # If the batch_size attribute of the instance is None, process entire dataset at once.
        if self.batch_size is None:
            self.batch_size = len(self.X_train_set)

        clusters = np.zeros(len(X_query_set))
        cluster_centers = torch.FloatTensor(self.cluster_centers_).to(device=self.device)

        idx = 0
        for i, data in tqdm(enumerate(X_query_loader),leave=False,disable=True if not self.verbose else False):
            # Compute the cluster label
            dist_mat = self.dist_fun(data.to(self.device),cluster_centers,**(self.metric_params if self.metric_params is not None else {}))
            clusters[idx:idx+len(data)] = np.argmin(dist_mat,axis=1)
            idx = idx+len(data)

        return clusters.astype(int)

    def _pairwise(self,x,y):
        if x.shape != y.shape:
            raise ValueError('x and y need to be the same shape')
        return np.diag(self.dist_fun(x,y))

    def fit_predict(self,X, y=None):
        """Performs fit (MiniBatchKMeans algorithm with settings passed during init) and predict (predicts for data of
        same dimension as training data to which cluster center its points belong) on the data X.

        Parameters

        :parameter X: array-like or h5py file handle.
                Training Data of shape (n_samples, n_features)

        Returns

        :return clusters: array of integers
                The cluster centers of the points, of shape (n_samples,)
        """
        return self.fit(X,y=y).predict(X)
