import cudf
from cuml.neighbors import NearestNeighbors as NearestNeighborsCUML
from cuml.cluster import KMeans as KMeansCUML
from cuml.decomposition import PCA as PCACUML

from time import time
import h5py as h5
import numpy as np
import pandas as pd
import os

DATA_PATH = '.'

def test_in_core_gpu(dataset,queryset):
    max_patients = int(1e5)
    X = dataset['X'][:max_patients]
    X_query = queryset['X'][:]

    patients_step = [10,100]#,1000,10000,20000,30000,40000,60000,80000,100000]
    out_df = pd.DataFrame(index=patients_step,columns=['neighbors_cuml','kmeans_cuml','pca_cuml'])

    # test neighbors
    print('Testing neighbors')
    for num_patients in patients_step:
        print(num_patients)
        X_step = X[:num_patients]
        batch_size = min(num_patients,1000)

        X_step_cudf = cudf.DataFrame(X_step)
        X_query_cudf = cudf.DataFrame(X_query)
        tic = time()
        neigh = NearestNeighborsCUML(n_neighbors=4)#,algorithm='brute')
        neigh.fit(X_step_cudf)
        dist_sk, neighb_sk = neigh.kneighbors(X_query_cudf, return_distance=True)
        out_df.loc[num_patients,['neighbors_cuml']] = time()-tic

    # test kmeans
    print('Testing KMeans')
    for num_patients in patients_step:
        print(num_patients)
        X_step = X[:num_patients]
        batch_size = min(num_patients,1000)
        init = X_step[:4]

        X_step_cudf = cudf.DataFrame(X_step)
        tic = time()
        kmeans_sk = KMeansCUML(n_clusters=4,n_init=1,init=init,random_state=47)
        kmeans_sk.fit(X_step_cudf)
        out_df.loc[num_patients,['kmeans_cuml']] = time()-tic

    # test pca
    print('Testing PCA')
    for num_patients in patients_step:
        print(num_patients)
        X_step = X[:num_patients]
        batch_size = min(num_patients,1000)

        X_step_cudf = cudf.DataFrame(X_step)
        tic = time()
        pca_sk = PCACUML(n_components=4)
        pca_sk.fit(X_step_cudf)
        out_df.loc[num_patients,['pca_cuml']] = time()-tic

    return out_df


if __name__=='__main__':
    dataset = 'snp'

    hf = h5.File(os.path.join(DATA_PATH,f'{dataset}_dataset.hdf5'),'r')
    hf_query = h5.File(os.path.join(DATA_PATH,f'{dataset}_queryset.hdf5'),'r')

    runtime_df_gpu = test_in_core_gpu(hf,hf_query)
    runtime_df_gpu.to_csv('runtime_df_gpu_cuml.csv')
