# Author: Michael Adamer
#         Eljas Roellin
#         Lucie Bourguignon
#
# License: BSD 3 clause

import torch
from torch.utils.data import DataLoader, RandomSampler

import numpy as np
from tqdm import tqdm

from simbsig.utils.datasets import arraysDataset,hdf5Dataset
from simbsig.utils.metrics import DistanceMetrics

# Alternative
# from scipy.linalg import qr, svd

class PCA:
    """Principal Component Analysis class. Implements Halko's algorithm[1], batched data loading for big datasets and
    optional GPU accelerated computations.

    Parameters

    :parameter n_components: int, default=None
            Number of principle components to be kept.
    :parameter iterated power: int, default=0
            The i in Halko's paper
    :parameter n_oversamples: int, default=n_components+2.
            The l in Halko's paper
    :parameter centered: bool, default=False
            Whether the features of the input data have been centered.
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
    :parameter verbose: bool, default=True
        Logging information. If True, progression updates are produced.

    [1] Halko, Nathan, et al. "An algorithm for the principal component analysis of large data sets." SIAM Journal
    on Scientific computing 33.5 (2011): 2580-2594.
    """

    def __init__(self, n_components=None, iterated_power=0, n_oversamples=None, centered=False,
                 device='cpu', mode='arrays', n_jobs=0, batch_size=None, random_state=None, verbose=True, **kwargs):

        self.n_components_ = n_components
        self.int_i = iterated_power # this is the i in Halko's paper
        self.int_l = n_oversamples if n_oversamples is not None else n_components + 2 # the l in Halko's paper, we default to heuristics
        self.centered = centered
        self.n_jobs = n_jobs #For dataloaders!
        self.mode = mode
        self.batch_size = batch_size
        self.device = torch.device('cuda') if device == 'gpu' else torch.device('cpu')
        self.verbose = verbose
        self.X_path = kwargs.pop('X_path', 'X')
        self.y_path = kwargs.pop('y_path', 'y')

        if random_state is not None:
            torch.manual_seed(random_state) # seed (usually not needed)

    def fit(self,X,y=None):
        """Performs principle component decomposition using Halko's method

        Parameters

        :parameter X: array-like or h5py file handle.
                Training Data of shape (n_samples, n_features) or \
                (n_samples, n_samples) if `metric='precomputed'`
        :parameter y: Ignored.
                Only present by convention.

        Returns

        :return self: PCA
                The PCA object with computed principal components
        """
        path_dict = {'X_path':self.X_path,'y_path':self.y_path}
        X_train_set = eval(f'{self.mode}Dataset(X,**path_dict)')
        self.n_features_in_ = X_train_set.n_features()

        # If n_components > n_features, raise valueerror
        if self.n_components_ > self.n_features_in_:
            raise ValueError(f"n_components={self.n_components_} have to be between 0 and the number of features"
                             f" ({self.n_features_in_}) in your dataset")

        # If the batch_size attribute of the instance is None, process entire dataset at once.
        if self.n_components_ is None:
            self.n_components_ = self.n_features_in_

        # If the batch_size attribute of the instance is None, process entire dataset at once.
        if self.batch_size is None:
            self.batch_size = len(X_train_set)
        else:
            self.batch_size = min(self.batch_size,len(X_train_set))

        X_train_loader = DataLoader(X_train_set,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.n_jobs)

        G = torch.randn(self.n_features_in_,self.int_l,device=self.device)
        H = []

        self.mean_ = torch.zeros(self.n_features_in_,dtype=torch.float).to(self.device)

        if not self.centered:
            for data in tqdm(X_train_loader,desc=f'Calculate mean',leave=False,disable=True if not self.verbose else False):
                self.mean_ += data.to(self.device).sum(dim=0)
            self.mean_ /= len(X_train_set)

        for i in range(self.int_i+1):
            H_parts = []
            for data in tqdm(X_train_loader,desc=f'Computing H matrix {i}/{self.int_i}',leave=False,disable=True if not self.verbose else False):
                data = (data.to(self.device)-self.mean_)
                if i == 0:
                    H_parts.append(torch.matmul(data,G))
                else:
                    H_subparts = 0.
                    for i,data_T in tqdm(enumerate(X_train_loader),desc=f'Computing A^TH matrix {i}/{self.int_i}',leave=False,disable=True if not self.verbose else False):
                        data_T = (data_T.to(self.device)-self.mean_).permute((1,0))
                        H_subparts += torch.matmul(data_T,H[-1][i*self.batch_size:(i+1)*self.batch_size])
                    H_parts.append(torch.matmul(data,H_subparts))
            H.append(torch.concat(H_parts,dim=0))
        H = torch.concat(H,dim=1)

        Q,R = torch.linalg.qr(H,mode='reduced')

        T = 0.
        for i,data_T in tqdm(enumerate(X_train_loader),desc='Computing T=A^TQ matrix',leave=False,disable=True if not self.verbose else False):
            data_T = (data_T.to(self.device)-self.mean_).permute((1,0))
            T += torch.matmul(data_T,Q[i*self.batch_size:(i+1)*self.batch_size])

        V,S,W_T = torch.linalg.svd(T,full_matrices=True)
        #U = torch.matmul(Q,W_T.permute((0,1)))

        # Store fitted singular_values_
        self.singular_values_ = S[:self.n_components_].cpu().numpy()

        # Store explained variance of each computed PC
        # self.explained_variance_ = np.power(self.singular_values_, 2) / max((len(X_train_set) - 1), 1)

        # Store the fitted PC's
        self.components_ = V[:,:self.n_components_].cpu().numpy()

        self.mean_ = self.mean_.cpu().numpy()
        return self

    def transform(self,X,centered=False):
        """Transforms data of same dimension as training data into dimension of n_components using
        the principal components computed during fit.

        Parameters

        :parameter X: array-like or h5py file handle.
                Training Data of shape (n_samples, n_features)
        :parameter centered: bool, default=False
                Whether the features of the input data have been centered.

        Returns

        :return X_transformed: torch.tensor
                The transformed data.
        """
        path_dict = {'X_path': self.X_path, 'y_path': self.y_path}
        # X_test_set = eval(f'{self.mode}Dataset(X=X)')
        X_test_set = eval(f'{self.mode}Dataset(X,**path_dict)')

        # Choose the appropriate batch size.
        batch_size = min(self.batch_size,len(X_test_set))

        X_test_loader = DataLoader(X_test_set,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=self.n_jobs)

        # Compute mean per feature
        # test_mean = torch.zeros(self.n_features_in_,dtype=torch.float).to(self.device)
        # if not centered:
        #     for data in tqdm(X_test_loader,desc=f'Calculate mean',leave=False,disable=True if not self.verbose else False):
        #         test_mean += data.to(self.device).sum(dim=0)
        #     test_mean /= len(X_test_set)
        #
        # test_mean = test_mean.cpu()

        out = []
        W_L = torch.tensor(self.components_).to(self.device)

        mean = torch.tensor(self.mean_).to(self.device)
        # Load data batchwise.
        idx=0
        for data in tqdm(X_test_loader,desc=f'Transforming',leave=False,disable=True if not self.verbose else False):
            # centering: subtract previously computed feature mean from test data
            if not centered:
                data = data.to(self.device) - mean#test_mean
            out.append(torch.matmul(data.to(self.device),W_L))
            idx += len(data)
        return torch.concat(out, axis=0).cpu().numpy()

    def fit_transform(self,X):
        """Performs fit (principle component decomposition using Halko's method) and transform (Transforms data of same
        data into dimension of n_components using the principal components computed during fit) on the data X.

        Parameters

        :parameter X: array-like or h5py file handle.
                Training Data of shape (n_samples, n_features)

        Returns

        :return X_transformed: torch.tensor
                The transformed data.
        """
        X_transformed = self.fit(X).transform(X)

        return X_transformed
