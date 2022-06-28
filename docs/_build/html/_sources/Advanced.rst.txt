Advanced
========

Custom Metric
-------------
SIMBSIG allows custom distance metrics to be used during similarity searches.

General Interface
^^^^^^^^^^^^^^^^^
These distance metrics should follow this interface:
   
.. code-block:: python

	def custom_metric(x1, x2, feature_weights=None, metric_params=None):
	    """Generic pairwise distance function 
            Parameters:

            :parameter x1: torch.tensor of dimension (n_samples, n_features)
            :parameter x2: torch.tensor of dimension (m_samples, n_features)
            :parameter feature_weights: torch.tensor of dimension (n_features,)
            :parameter metric_params: can be any parameter which which is handed over to the SIMBSIG neighbors module
            as metric_params
        
            Returns:

            :return dist_mat: numpy.array of dimension (n_samples, m_samples)

            Notice that n_samples does not have to be equal to m_samples. However, both n_features have to match.
            If GPU is available and a SIMBSIG neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
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
    
Example
^^^^^^^

As an example, we show how the 1 - rbf-kernel similarity could be used as custom distance metric for kernelised simliarity searches.
To use this for example in NearestNeighbors, the class instantiation should include `sigma` as key in a dictionary passed to `metric_params`:

.. code-block:: python


	nn_simbsig = NearestNeighbors(n_neighbors=n_neighbors, metric=custom_rbf_metric, metric_params={'sigma':2})
	
	

With the following example custom metric:


.. code-block:: python

    def custom_rbf_metric(x1, x2, p=None, feature_weights=None, sigma=None):
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

	    # First step: compute pairwise euclidean distances
	    euclidean_dist_mat = torch.pow(torch.cdist(x1, x2, 2), 2)

	    # Second step: exp(-euclidean_distance/sigma)
	    rbf_pairwise = torch.exp(-euclidean_dist_mat / sigma)
	    # dist_mat = 1 - rbf_pairwise
	    dist_mat = 1 - rbf_pairwise

	    # 2. Move the result off of the tensor, and convert to numpy.array
	    dist_mat = dist_mat.cpu().numpy()

	    # 3. return the dist_mat
	    return dist_mat

