import unittest
import numpy as np
import pathlib
import h5py
import os
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as KMeans_sk
from sklearn.cluster import MiniBatchKMeans as MinibatchKMeans_sk

from simbsig.cluster import MiniBatchKMeans

# Here are tests for the MiniBatchKMeans implementation.

class Test_MiniBatchKMeans_sklearn_simbsig(unittest.TestCase):

    def make_datasets(self):
        ''' Create datasets used for testing'''
        ################################################################################################################
        # Create Datasets
        N_ClUSTERS_LYS = []
        X_train_sets = []
        y_train_sets = []

        X_query_sets = []
        y_query_sets = []

        # 3 Clusters in 2 Dimensions, well separated
        N_CLUSTERS = 3
        TRN_PTS_PER_CLUSTER = 30
        TST_PTS_PER_CLUSTER = 10
        centers = [[-10, -10], [1, 2], [13, 8]]

        # Create Data
        X_train, y_train = make_blobs(n_samples=N_CLUSTERS * TRN_PTS_PER_CLUSTER, centers=centers, n_features=2,
                                      random_state=0)
        X_query, y_query = make_blobs(n_samples=N_CLUSTERS * TST_PTS_PER_CLUSTER, centers=centers, n_features=2,
                                      random_state=1)

        # Append it to list
        N_ClUSTERS_LYS.append(N_CLUSTERS)
        X_train_sets.append(X_train)
        y_train_sets.append(y_train)
        X_query_sets.append(X_query)
        y_query_sets.append(y_query)

        # 10 Clusters in 8 Dimensions, not necessarily well separated
        N_CLUSTERS = 8
        TRN_PTS_PER_CLUSTER = 100
        TST_PTS_PER_CLUSTER = 30
        N_DIM = 10
        rng = np.random.RandomState(42)
        centers = rng.uniform(low=-20,high=20,size=(N_CLUSTERS,N_DIM))

        # Create Data
        X_train, y_train = make_blobs(n_samples=N_CLUSTERS * TRN_PTS_PER_CLUSTER, centers=centers, n_features=N_DIM,
                                      random_state=42)
        X_query, y_query = make_blobs(n_samples=N_CLUSTERS * TST_PTS_PER_CLUSTER, centers=centers, n_features=N_DIM,
                                      random_state=43)

        # Append it to list
        N_ClUSTERS_LYS.append(N_CLUSTERS)
        X_train_sets.append(X_train)
        y_train_sets.append(y_train)
        X_query_sets.append(X_query)
        y_query_sets.append(y_query)

        return N_ClUSTERS_LYS, X_train_sets, y_train_sets, X_query_sets, y_query_sets


    def plot_predictions(self, X_test, y_pred, cluster_centers=None, title=None):
        '''Plot predictions, as visual aid for KMeans in 2D'''

        fig, ax = plt.subplots()
        ax.set_title("Small Synthetic Dataset")
        colors = ["#4EACC5", "#FF9C34", "#4E9A06"]
        colors_centers = ['b', 'm', 'r']
        for k, center in enumerate(cluster_centers):
            cluster_data = y_pred == k
            ax.scatter(X_test[cluster_data, 0], X_test[cluster_data, 1], c=colors[k], label=f"C{k}")

        ax.scatter(cluster_centers[:,0], cluster_centers[:,1], c=colors_centers)
        plt.title(title)
        ax.legend()

        plt.show()

    def test_MiniBatchKMeans_sklearn_simbsig(self):
        '''Compare simbsig.clustering.MiniBatchKMeans vs sklearn.clustering.KMeans as gold standard'''

        ###############################################################################################################
        # Make datasets
        N_ClUSTERS_LYS, X_train_sets, y_train_sets, X_query_sets, y_query_sets = self.make_datasets()

        ###############################################################################################################
        # Safe one dataset to hdf5 file format
        dataset_path = pathlib.Path(__file__).resolve().parents[0]
        train_file = f'train.hdf5'
        query_file = f'query.hdf5'

        with h5py.File(os.path.join(dataset_path, f"{train_file}"), 'w') as f:
             f.create_dataset("X",data=X_train_sets[0])

        with h5py.File(os.path.join(dataset_path, f"{query_file}"), 'w') as f:
             f.create_dataset("X",data=X_query_sets[0])

        # Load hdf5 files
        X_train_hdf5 = h5py.File(os.path.join(dataset_path, train_file), 'r')
        X_query_hdf5 = h5py.File(os.path.join(dataset_path, query_file))

        # n_reps is the number of repetitions (each individual repetition seeded differently) for which
        # these two stochastic methods are compared. an average over these repetitions is used as test statistic.
        n_reps = 20

        # INF as very large number
        INF = 1000000

        # Test multiple parameter combinations
        for i in range(len(X_train_sets)):
        #for i in [0]:
            # Load one dataset
            n_clusters = N_ClUSTERS_LYS[i]
            X_train = X_train_sets[i]
            y_train = y_train_sets[i]
            X_query = X_query_sets[i]
            y_query = y_query_sets[i]

            for metric in ['euclidean']: # sklearn does not support other metrics than euclidean
                for metric_params in [None]: # sklearn (euclidean only) does not accept other parameters
                    for feature_weights in [None]: # sklearn does not support other than uniform feature weights
                        for max_iter in [100]:
                            for tol in [1e-2]:
                               for device in ['cpu']:#,'gpu']:
                                # Validity check of hdf5 only for one dataset
                                    if n_clusters==3:
                                        mode_lys = ['arrays', 'hdf5']
                                    else:
                                        mode_lys = ['arrays']
                                    for mode in mode_lys:
                                        for batch_size in [1000, INF]:
                                            for init_type in ['fixed_centers']:
                                                for alpha in [0.95]:
                                                    for random_state in [len(X_train)]:
                                                        n_jobs=0

                                                        # use for each repetition another random state
                                                        rng = np.random.RandomState(random_state)
                                                        rand_state_rep_lys = rng.randint(batch_size,
                                                                                         batch_size*2,size=(n_reps,))

                                                        inf_string = f'dataset {i}' \
                                                                     f' n_clusters {n_clusters} metric {metric}' \
                                                                     f' metric_params {metric_params}' \
                                                                     f' feature_weights {feature_weights}' \
                                                                     f' max_iter {max_iter} tol {tol} ' \
                                                                     f' device: {device} mode: {mode}' \
                                                                     f' batch_size: {batch_size} ' \
                                                                     f' init_type {init_type} alpha {alpha}'

                                                        simbsig_sk_ratio = []
                                                        simbsig_sk_ratio_batch = []
                                                        for rep in range(n_reps):

                                                            if init_type == 'fixed_centers':
                                                                if n_clusters == 3:
                                                                    init = np.array([[-8, -8], [0, 0], [8, 8]])
                                                                elif n_clusters == 8:
                                                                    rng_2 = np.random.RandomState(rep)
                                                                    init = rng_2.uniform(
                                                                        low=-10, high=10, size=(n_clusters, 10))
                                                            elif init_type == 'random':
                                                                init = 'random'

                                                            ################################################################
                                                            # Instantiation

                                                            sk_KMeans = KMeans_sk(
                                                                n_clusters=n_clusters, init='random',
                                                                max_iter=max_iter, tol=tol,
                                                                algorithm='lloyd', random_state=rand_state_rep_lys[rep])

                                                            sk_MiniBatchKMeans = MinibatchKMeans_sk(
                                                                n_clusters=n_clusters,
                                                                random_state=rand_state_rep_lys[rep],
                                                                batch_size=batch_size,
                                                                init=init,
                                                                n_init=1
                                                            )

                                                            simbsig_MiniBatchKMeans = MiniBatchKMeans(
                                                                n_clusters=n_clusters,
                                                                metric=metric,
                                                                metric_params=metric_params,
                                                                feature_weights=feature_weights,
                                                                max_iter=max_iter, tol=tol,
                                                                device=device, mode=mode,
                                                                n_jobs=n_jobs,
                                                                batch_size=batch_size, init=init, # init=np.array([[0,0], [10,2], [-2,-2]]),
                                                                alpha=alpha,
                                                                random_state=rand_state_rep_lys[rep],
                                                                verbose=False)

                                                            ################################################################
                                                            # fit
                                                            sk_KMeans.fit(X_train)
                                                            sk_MiniBatchKMeans.fit(X_train)
                                                            if mode == 'arrays':
                                                                simbsig_MiniBatchKMeans.fit(X_train)
                                                            else: # mode == 'hdf5'
                                                                simbsig_MiniBatchKMeans.fit(X_train_hdf5)

                                                            ################################################################
                                                            # predict
                                                            sk_pred = sk_KMeans.predict(X_query)
                                                            sk_pred_batch = sk_MiniBatchKMeans.predict(X_query)

                                                            if mode == 'arrays':
                                                                simbsig_pred = simbsig_MiniBatchKMeans.predict(X_query)
                                                            else: # mode == 'hdf5'
                                                                simbsig_pred = simbsig_MiniBatchKMeans.predict(X_query_hdf5)

                                                            ################################################################
                                                            # compare residuals
                                                            sk_minibatch = np.linalg.norm(sk_MiniBatchKMeans.cluster_centers_- sk_KMeans.cluster_centers_)
                                                            simbsig_minibatch = np.linalg.norm(simbsig_MiniBatchKMeans.cluster_centers_- sk_KMeans.cluster_centers_)
                                                            simbsig_sk_ratio.append(simbsig_minibatch/sk_minibatch)
                                                            # if simbsig_minibatch>sk_minibatch+0.1:
                                                                # print(simbsig_minibatch,sk_minibatch,inf_string)
                                                            # self.assertTrue(simbsig_minibatch/sk_minibatch>1.2,
                                                            #                     msg=f'Frobenius norm of simbsig clusters to sklearn clusters is larger than sklearn MiniBatchKMeans to sklearn')
                                                        #     sk_residual = np.linalg.norm(
                                                        #         sk_KMeans.cluster_centers_[sk_pred] - X_query)
                                                        #     simbsig_residual = np.linalg.norm(
                                                        #         simbsig_MiniBatchKMeans.cluster_centers_[simbsig_pred] - X_query)
                                                        #
                                                        #     sk_residual_batch = np.linalg.norm(
                                                        #         sk_MiniBatchKMeans.cluster_centers_[sk_pred_batch] - X_query
                                                        #     )
                                                        #
                                                        #     simbsig_sk_ratio.append(simbsig_residual/sk_residual)
                                                        #     simbsig_sk_ratio_batch.append(simbsig_residual/sk_residual_batch)
                                                        #
                                                        #     # TODO: reactivate with reasonable threshold
                                                        #     '''
                                                        #     self.assertTrue(np.mean(simbsig_sk_ratio) > 0.9 and
                                                        #                     np.mean(simbsig_sk_ratio) < 1.1,
                                                        #                     msg=f'Average ratio of simbsig residuals'
                                                        #                         f' and sklearn residuals over {n_reps} runs'
                                                        #                         f' is {np.mean(simbsig_sk_ratio)}'
                                                        #                         f'for {inf_string}')
                                                        #     '''
                                                        #
                                                        self.assertTrue(np.mean(simbsig_sk_ratio)<1.1,
                                                                        msg=f'Average Frobenius norm of simbsig clusters to sklearn clusters is larger than sklearn MiniBatchKMeans to sklearn')
                                                        # print(f'avg ratio simbsig vs KMeans init_type {init_type}: {np.mean(simbsig_sk_ratio)}')
                                                        # print(f'avg ratio simbsig vs MiniBatchKMeans init_type {init_type}: {np.mean(simbsig_sk_ratio_batch)})')

        X_train_hdf5.close()
        X_query_hdf5.close()

        '''
        # Optional plotting guidance for 2D example
        self.plot_predictions(X_query, simbsig_pred,
                              simbsig_MiniBatchKMeans.cluster_centers_,
                              'simbsig')

        self.plot_predictions(X_query, sk_pred,
                              sk_KMeans.cluster_centers_, 'sklearn')
        '''

if __name__ == '__main__':
    unittest.main()
