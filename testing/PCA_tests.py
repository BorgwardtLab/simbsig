# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

import unittest
import numpy as np
import pathlib
import h5py
import os

from sklearn.decomposition import PCA as PCA_sk
from simbsig.decomposition import PCA

# Here are tests for the PCA implementation.

class Test_PCA_sklearn_simbsig(unittest.TestCase):
    def test_PCA_sklearn_simbsig(self):
        '''
        Compare simbsig.decomposition.PCA vs sklearn.decomposition.PCA as gold standard.
        Notice from Halko's Paper: suggest data matrix m x n, choosing k principle components
        -The method works best if k << min(m,n)
        -"For most practical applications, iterated_power <= 2 and n_oversamples=k+2 is sufficient"
        If k << min(m,n) is ambiguous, we make n_oversamples=k+6 rather than k+2, which is more memory and
        computationally
        expensive, but more accurate.
        '''

        ################################################################################################################
        # Create Dataset
        np.random.seed(98)
        n_samples = 100
        n_queries = 50
        n_dim = 20 # Tested for 10, 20
        X_train = np.random.uniform(low=-5, high=5, size=(n_samples, n_dim))

        X_query = np.random.uniform(low=-5, high=5, size=(n_queries, n_dim))

        # feature_weights vector
        w = np.random.uniform(size=(n_dim))

        ###############################################################################################################
        # Safe to hdf5 file format
        dataset_path = pathlib.Path(__file__).resolve().parents[0]
        train_file = f'train.hdf5'
        query_file = f'query.hdf5'

        with h5py.File(os.path.join(dataset_path, f"{train_file}"), 'w') as f:
             f.create_dataset("X",data=X_train)

        with h5py.File(os.path.join(dataset_path, f"{query_file}"), 'w') as f:
             f.create_dataset("X",data=X_query)

        # Load hdf5 files
        train_data = h5py.File(os.path.join(dataset_path, train_file), 'r')
        query_data = h5py.File(os.path.join(dataset_path, query_file))

        INF = 1000000
        for n_components in [1, 2, 5]:
            for iterated_power in [5]:
                for n_oversamples in [n_components+2]:
                    for centered in [False]:
                      for device in ['cpu']:#,'gpu']:
                        for mode in ['arrays', 'hdf5']:
                            for batch_size in [30, 62, INF]:
                                random_state=47

                                inf_string = f'n_components {n_components} iterated_power {iterated_power}' \
                                             f' n_oversamples {n_oversamples} centered {centered}' \
                                             f' device: {device} mode: {mode} batch_size: {batch_size} ' \
                                             f' random_state {random_state}'

                                ########################################################################################
                                # Instantiation

                                # Not necessarily exact
                                #sk_PCA = PCA_sk(n_components=n_components, svd_solver='randomized',
                                #                iterated_power=iterated_power, n_oversamples=n_oversamples,
                                #                random_state=random_state)

                                sk_PCA = PCA_sk(n_components)
                                sk_approx_PCA = PCA_sk(n_components,svd_solver='randomized',
                                                       random_state=random_state,
                                                       iterated_power=iterated_power,
                                                       n_oversamples=n_oversamples,)

                                simbsig_PCA = PCA(n_components=n_components, iterated_power=iterated_power,
                                             n_oversamples=n_oversamples, centered=centered, device=device, mode=mode,
                                             batch_size=batch_size, random_state=random_state,
                                             verbose=False)

                                ########################################################################################
                                # fit

                                if mode=='arrays':
                                    simbsig_PCA.fit(X_train)
                                else:
                                    simbsig_PCA.fit(train_data)
                                sk_PCA.fit(X_train)
                                sk_approx_PCA.fit(X_train)

                                ########################################################################################
                                # test if components are element-wise within tolerance up to sign, which may be
                                # different between different implementations. relative tolerance: 0.02
                                REL_TOL = 0.01
                                components_sk = sk_PCA.components_
                                components_simbsig = simbsig_PCA.components_.T

                                components_allclose = np.full(len(components_sk),False)
                                for i in range(len(components_sk)):

                                    # Test if for same-signed or opposite signed principal components,
                                    # the absolute difference of each components entry is smaller than the tolerance
                                    close_i = np.abs(np.abs(components_sk[i]) - np.abs(components_simbsig[i])) < REL_TOL

                                    # aggregate the individual element's tests to the entire component being
                                    # within the tolerance
                                    close = np.alltrue(close_i)

                                    components_allclose[i] = close

                                self.assertTrue(np.alltrue(components_allclose),
                                                msg=f'Failed PC comparison: {inf_string}')

                                ########################################################################################
                                # test if transformation is element-wise within tolerance up to sign, which may be
                                # different between different implementations. relative tolerance: 0.02
                                REL_TOL = 0.1
                                trafo_sk = sk_PCA.transform(X_train)
                                if mode=='arrays':
                                     trafo_simbsig = simbsig_PCA.transform(X_train, centered=False)
                                elif mode=='hdf5':
                                     trafo_simbsig = simbsig_PCA.transform(train_data, centered=False)

                                trafo_allclose = np.full(n_components,False)

                                for i in range(n_components):

                                    # Test if for same-signed or opposite signed principal components,
                                    # the absolute difference of each components entry is smaller than the tolerance
                                    close_i = np.abs(np.abs(trafo_sk[:,i]) - np.abs(trafo_simbsig[:,i]))/np.abs(trafo_sk[:,i]) < REL_TOL

                                    # aggregate the individual element's tests to the entire transformation's
                                    # component being within the tolerance
                                    close = np.alltrue(close_i)

                                    trafo_allclose[i] = close

                                self.assertTrue(np.alltrue(trafo_allclose),
                                                 msg=f'Failed transform comparison: {inf_string}')


                                ########################################################################################
                                # test singular values
                                sv_sk = sk_PCA.singular_values_
                                sv_approx_sk = sk_approx_PCA.singular_values_
                                sv_simbsig = simbsig_PCA.singular_values_
                                sk_random = np.linalg.norm(sk_PCA.singular_values_- sk_approx_PCA.singular_values_)
                                simbsig_random = np.linalg.norm(sk_PCA.singular_values_- simbsig_PCA.singular_values_)
                                self.assertTrue(simbsig_random<sk_random,
                                                msg=f'Failed singular value comparison: {inf_string}')
                                # self.assertTrue(np.allclose(sv_sk, sv_simbsig),
                                #                 msg=f'Failed singular value comparison: {inf_string}')

                                ########################################################################################
                                # test explained variance
                                # expl_var_sk = sk_PCA.explained_variance_
                                # expl_var_simbsig = simbsig_PCA.explained_variance_
                                # self.assertTrue(np.allclose(expl_var_sk, expl_var_simbsig),
                                #                 msg=f'Failed variance explained comparison: {inf_string}')

        train_data.close()
        query_data.close()

if __name__ == '__main__':
    unittest.main()
