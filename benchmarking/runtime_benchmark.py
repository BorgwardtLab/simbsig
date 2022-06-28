from bigsise.neighbors import NearestNeighbors as NearestNeighborsOwn
from bigsise.clustering import MiniBatchKMeans as MiniBatchKMeansOwn
from bigsise.decomposition import PCA as PCAOwn

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

# import cudf
# from cuml.neighbors import NearestNeighbors as NearestNeighborsCUML
# from cuml.cluster import KMeans as KMeansCUML
# from cuml.decomposition import PCA as PCACUML


from time import time
import h5py as h5
import numpy as np
import pandas as pd
import os

from multiprocessing import cpu_count

DATA_PATH = '/home/michael/ETH/data/BIGSISE'


def test_in_core_cpu(dataset,queryset):
    X_query = queryset['X'][:]

    patients_step = [10,100,1000,10000,20000,30000,40000,60000,80000,100000]
    out_df = pd.DataFrame(index=patients_step,columns=['neighbors_sklearn','neighbors_bigsise','kmeans_sklearn','kmeans_bigsise','pca_sklearn','pca_bigsise'])

    # test neighbors
    print('Testing neighbors')
    for num_patients in patients_step:
        print(num_patients)
        X_step = dataset['X'][:num_patients]
        batch_size = min(num_patients,60000)

        tic = time()
        neigh = NearestNeighbors(n_neighbors=4)#,algorithm='brute')
        neigh.fit(X_step)
        dist_sk, neighb_sk = neigh.kneighbors(X_query, return_distance=True)
        out_df.loc[num_patients,['neighbors_sklearn']] = time()-tic

        tic = time()
        neigh_own = NearestNeighborsOwn(n_neighbors=4,metric='euclidean')
        neigh_own.fit(X_step)
        dist_own, neighb_own = neigh_own.kneighbors(X_query, return_distance=True)
        out_df.loc[num_patients,['neighbors_bigsise']] = time()-tic

    # test kmeans
    print('Testing KMeans')
    for num_patients in patients_step:
        print(num_patients)
        X_step = dataset['X'][:num_patients]
        batch_size = min(num_patients,10000)
        init = X_step[:4]

        tic = time()
        kmeans_sk = MiniBatchKMeans(n_clusters=4,n_init=1,init=init,random_state=47,
                                    batch_size=batch_size)#,tol=1e-4,max_no_improvement=None,reassignment_ratio=0.0)
        kmeans_sk.fit(X_step)
        out_df.loc[num_patients,['kmeans_sklearn']] = time()-tic

        # kmeans_sk_ex = KMeans(n_clusters=4,n_init=1,init=init,random_state=47)
        # kmeans_sk_ex.fit(X_step)

        tic = time()
        kmeans_bigsise = MiniBatchKMeansOwn(n_clusters=4,init=init,random_state=47,batch_size=batch_size)
        kmeans_bigsise.fit(X_step)
        out_df.loc[num_patients,['kmeans_bigsise']] = time()-tic

        # sort_idx_sk = np.argsort(kmeans_sk.cluster_centers_[:,0])
        # sort_idx_sk_ex = np.argsort(kmeans_sk_ex.cluster_centers_[:,0])
        # sort_idx_bigsise = np.argsort(kmeans_bigsise.cluster_centers_[:,0])
        # print(kmeans_sk_ex.cluster_centers_[sort_idx_sk_ex])
        # print(kmeans_sk.cluster_centers_[sort_idx_sk])
        # print(kmeans_bigsise.cluster_centers_[sort_idx_bigsise])
        #
        # print(np.linalg.norm(kmeans_sk_ex.cluster_centers_[sort_idx_sk_ex]-kmeans_sk.cluster_centers_[sort_idx_sk]))
        # print(np.linalg.norm(kmeans_sk_ex.cluster_centers_[sort_idx_sk_ex]-kmeans_bigsise.cluster_centers_[sort_idx_bigsise]))

    # test pca
    print('Testing PCA')
    for num_patients in patients_step:
        print(num_patients)
        X_step = dataset['X'][:num_patients]
        batch_size = min(num_patients,10000)

        tic = time()
        pca_sk = PCA(n_components=4,svd_solver='randomized',iterated_power=0,n_oversamples=6,random_state=47)
        pca_sk.fit(X_step)
        out_df.loc[num_patients,['pca_sklearn']] = time()-tic

        tic = time()
        pca_bigsise = PCAOwn(n_components=4,iterated_power=0,n_oversamples=6,random_state=47,batch_size=batch_size)
        pca_bigsise.fit(X_step)
        out_df.loc[num_patients,['pca_bigsise']] = time()-tic

        # print(pca_bigsise.singular_values_)

    return out_df

def test_in_core_gpu(dataset,queryset):
    X_query = queryset['X'][:]

    patients_step = [10,100,1000,10000,20000,30000,40000,60000,80000,100000]
    out_df = pd.DataFrame(index=patients_step,columns=['neighbors_cuml','neighbors_bigsise','kmeans_cuml','kmeans_bigsise','pca_cuml','pca_bigsise'])

    # test neighbors
    print('Testing neighbors')
    for num_patients in patients_step:
        print(num_patients)
        X_step = dataset['X'][:num_patients]
        batch_size = min(num_patients,60000)

        # X_step_cudf = cudf.DataFrame(X_step)
        # X_query_cudf = cudf.DataFrame(X_query)
        # tic = time()
        # neigh = NearestNeighborsCUML(n_neighbors=4)#,algorithm='brute')
        # neigh.fit(X_step_cudf)
        # dist_sk, neighb_sk = neigh.kneighbors(X_query_cudf, return_distance=True)
        # out_df.loc[num_patients,['neighbors_cuml']] = time()-tic

        tic = time()
        neigh_own = NearestNeighborsOwn(n_neighbors=4,metric='euclidean',device='gpu',batch_size=batch_size)
        neigh_own.fit(X_step)
        dist_own, neighb_own = neigh_own.kneighbors(X_query, return_distance=True)
        out_df.loc[num_patients,['neighbors_bigsise']] = time()-tic

    # test kmeans
    print('Testing KMeans')
    for num_patients in patients_step:
        print(num_patients)
        X_step = dataset['X'][:num_patients]
        batch_size = min(num_patients,10000)
        init = X_step[:4]

        # X_step_cudf = cudf.DataFrame(X_step)
        # tic = time()
        # kmeans_sk = KMeansCUML(n_clusters=4,n_init=1,init=init,random_state=47)
        # kmeans_sk.fit(X_step_cudf)
        # out_df.loc[num_patients,['kmeans_cuml']] = time()-tic

        tic = time()
        kmeans_bigsise = MiniBatchKMeansOwn(n_clusters=4,init=init,random_state=47,device='gpu',batch_size=batch_size)
        kmeans_bigsise.fit(X_step)
        out_df.loc[num_patients,['kmeans_bigsise']] = time()-tic

    # test pca
    print('Testing PCA')
    for num_patients in patients_step:
        print(num_patients)
        X_step = dataset['X'][:num_patients]
        batch_size = min(num_patients,10000)

        # X_step_cudf = cudf.DataFrame(X_step)
        # tic = time()
        # pca_sk = PCACUML(n_components=4)
        # pca_sk.fit(X_step_cudf)
        # out_df.loc[num_patients,['pca_cuml']] = time()-tic

        tic = time()
        pca_bigsise = PCAOwn(n_components=4,device='gpu',iterated_power=0,n_oversamples=6,batch_size=batch_size,random_state=47)
        pca_bigsise.fit(X_step)
        out_df.loc[num_patients,['pca_bigsise']] = time()-tic

    return out_df

def test_out_of_core(dataset,queryset):
    patients_step = [10,100,1000,10000,20000,30000,40000,60000,80000,100000,200000,300000,400000,500000]
    out_df = pd.DataFrame(index=patients_step,columns=['neighbors_ooc_cpu','neighbors_ooc_gpu','kmeans_ooc_cpu','kmeans_ooc_gpu','pca_ooc_cpu','pca_ooc_gpu'])

    X_query = h5.File(os.path.join(DATA_PATH,f'{dataset}_queryset.hdf5'),'r')


    for device in ['cpu','gpu']:
        print(device.upper()+':')

        n_jobs = cpu_count()# if device == 'gpu' else cpu_count()-1

        # test neighbors
        print('Testing neighbors')
        for num_patients in patients_step:
            print(num_patients)
            X = h5.File(os.path.join(DATA_PATH,f'{dataset}_dataset_{num_patients}.hdf5'),'r')
            batch_size = 1000#min(num_patients,1000)

            tic = time()
            neigh_own = NearestNeighborsOwn(n_neighbors=4,metric='euclidean',mode='hdf5',device=device,batch_size=batch_size,n_jobs=n_jobs)
            neigh_own.fit(X)
            dist_own, neighb_own = neigh_own.kneighbors(X_query, return_distance=True)
            out_df.loc[num_patients,[f'neighbors_ooc_{device}']] = time()-tic

        # test kmeans
        print('Testing KMeans')
        for num_patients in patients_step:
            print(num_patients)
            X = h5.File(os.path.join(DATA_PATH,f'{dataset}_dataset_{num_patients}.hdf5'),'r')
            batch_size = min(num_patients,10000)
            init = X['X'][:4]

            tic = time()
            kmeans_bigsise = MiniBatchKMeansOwn(n_clusters=4,init=init,random_state=47,mode='hdf5',device=device,batch_size=batch_size,n_jobs=n_jobs)
            kmeans_bigsise.fit(X)
            out_df.loc[num_patients,[f'kmeans_ooc_{device}']] = time()-tic

            # sort_idx_bigsise = np.argsort(kmeans_bigsise.cluster_centers_[:,0])
            # print(kmeans_bigsise.cluster_centers_[sort_idx_bigsise])

        # test pca
        print('Testing PCA')
        for num_patients in patients_step:
            print(num_patients)
            X = h5.File(os.path.join(DATA_PATH,f'{dataset}_dataset_{num_patients}.hdf5'),'r')
            batch_size = min(num_patients,10000)

            tic = time()
            pca_bigsise = PCAOwn(n_components=4,mode='hdf5',device=device,batch_size=batch_size,n_jobs=n_jobs)
            pca_bigsise.fit(X)
            out_df.loc[num_patients,[f'pca_ooc_{device}']] = time()-tic

            # print(pca_bigsise.singular_values_)

    return out_df

if __name__=='__main__':
    n_runs = 10
    dataset = 'snp'

    hf = h5.File(os.path.join(DATA_PATH,f'{dataset}_dataset.hdf5'),'r')
    hf_query = h5.File(os.path.join(DATA_PATH,f'{dataset}_queryset.hdf5'),'r')

    for run in range(n_runs):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Run {run+1} of {n_runs}:')

        print('In core CPU')
        runtime_df_cpu = test_in_core_cpu(hf,hf_query)
        runtime_df_cpu.to_csv(os.path.join(DATA_PATH,f'runtime_df_cpu_{run}.csv'))
        # print(runtime_df_cpu)

        print('In core GPU')
        runtime_df_gpu = test_in_core_gpu(hf,hf_query)
        runtime_df_gpu.to_csv(os.path.join(DATA_PATH,f'runtime_df_gpu_{run}.csv'))
        # print(runtime_df_gpu)

        print('Out of core')
        runtime_df_ooc = test_out_of_core(dataset,hf_query)
        runtime_df_ooc.to_csv(os.path.join(DATA_PATH,f'runtime_df_ooc_{run}.csv'))
        # print(runtime_df_ooc)
