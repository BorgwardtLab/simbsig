# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

import unittest

from testing.NearestNeighbors_tests import Test_NearestNeighbors, Test_NearestNeighbors_sklearn_simbsig
from testing.KNeighborsClassifier_tests import Test_KNeighborsClassifier, Test_KNeighborsClassifier_sklearn_simbsig
from testing.KNeighborsRegressor_tests import Test_KNeighborsRegressor, Test_KNeighborsRegressor_sklearn_simbsig
from testing.RadiusNeighborsClassifier_tests import Test_RadiusNeighborsClassifier_sklearn_simbsig
from testing.RadiusNeighborsRegressor_tests import Test_RadiusNeighborsRegressor_sklearn_simbsig
from testing.Precomputed_tests import Test_metric_precomputed
from testing.PCA_tests import Test_PCA_sklearn_simbsig
from testing.MiniBatchKMeans_tests import Test_MiniBatchKMeans_sklearn_simbsig
from testing.Callable_metric_tests import Test_metric_callable

# This is script collectively runs the tests of Bigsise's individual modules.

# At the moment, we only test on CPU as the algorithms use the same code on CPU and GPU.
# CUDA seems seems to produce marginally different outputs as CPU computations at times.
# We attribute this to the different routines pytorch uses for CPU and CUDA computations.

if __name__ == '__main__':
    unittest.main()
