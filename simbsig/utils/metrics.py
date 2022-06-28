# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

import torch
import numpy as np

class DistanceMetrics:

    def __init__(self, device='cpu'):
        self.device = device

    def _minkowski(self, x1, x2, p=1, feature_weights=None):
        if feature_weights is not None:
            feature_weights = torch.pow(feature_weights, 1.0 / p)
            x1 = feature_weights*x1
            x2 = feature_weights*x2

        result = torch.cdist(x1, x2, p)

        return result.cpu().numpy()

    def _manhattan(self, x1, x2, p=1, feature_weights=None):
        return self._minkowski(x1, x2, p=1, feature_weights=feature_weights)

    def _euclidean(self, x1, x2, p=2, feature_weights=None):
        return self._minkowski(x1, x2, p=2, feature_weights=feature_weights)

    def _hamming(self, x1, x2, p=0):
        return self._minkowski(x1, x2, p=0)

    def _fractional(self, x1, x2, p=1, feature_weights=None):
        return np.power(self._minkowski(x1, x2, p=p, feature_weights=feature_weights), 1./p)

    def _cosine(self, x1, x2, p=None, feature_weights=None):
        if feature_weights is not None:
            feature_weights = torch.pow(feature_weights, 0.5)
            x1 = feature_weights*x1
            x2 = feature_weights*x2

        x1_normalised = torch.nn.functional.normalize(x1, dim=1)
        x2_normalised = torch.nn.functional.normalize(x2, dim=1)

        res = 1-torch.matmul(x1_normalised, x2_normalised.T)
        # res *= -1  # 1-res without copy
        # res += 1

        return res.cpu().numpy()

    def _mahalanobis(self, x1, x2, p=None, feature_weights=None, VI=None):

        if VI is None:
            raise ValueError('No inverse covariance matrix given.')
        else:
            VI = torch.tensor(VI, dtype=torch.float32).to(self.device)

        if feature_weights is not None:
            feature_weights = feature_weights.to(self.device)
            x1 = feature_weights*x1
            x2 = feature_weights*x2

        result = torch.zeros((len(x1), len(x2)))
        for i in range(len(x1)):
            delta = x2 - x1[i,:].unsqueeze(0)
            # Try this:
            result[i,:] = (torch.mm((delta), VI) * delta).sum(dim=1)
            #torch.diag(torch.mm(torch.mm((delta), VI), delta.T))

        result = torch.sqrt(result)

        return result.cpu().numpy()

    def _mahalanobis_init(self,X_loader,feature_weights=None,full=True,device='cpu'):
        '''
           If full is true then compute the full inverse
           else compute the diagonal elements only
        '''
        if device == 'gpu':
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        if feature_weights is not None:
            feature_weights = torch.tensor(feature_weights).to(device)

        # Loop over dataset first to get the mean
        mean = None
        for x in X_loader:
            x = x.to(device)
            if feature_weights is not None:
                x = feature_weights*x
            if mean is None:
                mean = x.sum(dim=0)
            else:
                mean += x.sum(dim=0)
        mean = mean/len(X_loader.dataset)

        S = torch.zeros(mean.shape[0],mean.shape[0],device=device)

        for x in X_loader:
            x = x.to(device)
            if feature_weights is not None:
                x = feature_weights*x

            S += torch.matmul((x-mean).unsqueeze(1),(x-mean).unsqueeze(1).permute(0,2,1)).sum(dim=0)

        S = S/(len(X_loader.dataset)-1)

        return torch.inverse(S).cpu().numpy()


def custom_metric(x1, x2, feature_weights=None, custom_params=None):
    """Generic pairwise distance function
    Parameters:

    :parameter x1: torch.tensor of dimension (n_samples, n_features)
    :parameter x2: torch.tensor of dimension (m_samples, n_features)
    :parameter feature_weights: torch.tensor of dimension (n_features,)
    :parameter custom_params: passed as metric_params={'custom_param_1':..., 'custom_param_2':...} in constructor.
    Any custom parameter name may be chosen.

    Returns:

    :return dist_mat: numpy.array of dimension (n_samples, m_samples)

    Notice that n_samples does not have to be equal to m_samples. However, both n_features have to match.
    If GPU is available and a simbsig neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
    RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
    be handed over to custom_metric the GPU.

    """

    # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
    # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
    # np.array operations is possible. Optionally, feature weights can be used.

    # dist_mat = some_operations(x1, x2)

    # 2. Move the result off of the tensor, and convert to numpy.array
    # dist_mat = dist_mat.cpu().numpy

    # 3. return the dist_mat
    # return dist_mat
    pass



def rbf_metric(x1, x2, p=None, feature_weights=None, sigma=None):
    """Example pairwise distance function
    Parameters:

    :parameter x1: torch.tensor of dimension (n_samples, n_features)
    :parameter x2: torch.tensor of dimension (m_samples, n_features)
    :parameter feature_weights: torch.tensor of dimension (n_features,)
    :parameter sigma: passed as metric_params={'sigma':int} in constructor. any custom parameter name may be
    chosen.

    Returns:

    :return dist_mat: numpy.array of dimension (n_samples, m_samples)

    Notice that n_samples does not have to be equal to m_samples. However, both n_features have to match.
    If GPU is available and a simbsig neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
    RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
    be handed over to custom_metric the GPU.

    """

    # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
    # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
    # np.array operations is possible. Optionally, feature weights can be used.

    # First step: compute pairwise squared distances
    squared_dist_mat = torch.pow(torch.cdist(x1, x2, 2), 2)

    # Second step: exp(-squared_dist_mat/sigma)
    rbf_pairwise = torch.exp(-squared_dist_mat / sigma)
    # dist_mat = 1 - rbf_pairwise
    dist_mat = 1 - rbf_pairwise

    # 2. Move the result off of the tensor, and convert to numpy.array
    dist_mat = dist_mat.cpu().numpy()

    # 3. return the dist_mat
    return dist_mat
