import os
import numpy as np
import h5py as h5

from tqdm import tqdm

DATA_PATH = '.'

def generate_snps():
    num_patients = int(1e3)
    num_snps = int(1e4)
    snp_step = int(1e3)

    snps = np.array([0,1,2],dtype='i2')


    with h5.File(os.path.join(DATA_PATH,'snp_queryset.hdf5'),'w') as hf:
        x_set = hf.create_dataset('X',(num_patients,num_snps),dtype='i2')#,compression='lzf')
        for i in tqdm(np.arange(0,num_snps,snp_step)):
            data = np.random.choice(snps,p=[0.6,0.2,0.2],size=(num_patients,snp_step)).astype(np.int8)
            x_set[:,i:i+snp_step] = data

    # Test the file
    with h5.File(os.path.join(DATA_PATH,'snp_queryset.hdf5'),'r') as hf:
        print(hf['X'].shape)
        print(hf['X'][:10,:100])

def generate_gaussian_mixture():
    num_components = 4
    num_patients = int(1e3)
    # if num_patients % num_components == 0:
    #     patients_per_cmponent = num_patients/num_components
    # else:
    #     raise ValueError('Number of patients needs to be divisible by number of components.')

    num_features = int(1e4)
    features_step = int(1e3)

    means = np.zeros((num_patients,num_components))
    component = np.random.randint(low=0,high=num_components-1,size=(num_patients))

    for i,m in tqdm(enumerate(means)):
        m[component[i]] = 1

    # cov = np.eye(num_features)

    with h5.File(os.path.join(DATA_PATH,'mixture_queryset.hdf5'),'w') as hf:
        x_set = hf.create_dataset('X',(num_patients,num_features))#,compression='lzf')
        for i in tqdm(np.arange(0,num_features,features_step)):
            x_set[:,i:i+features_step] = np.random.randn(num_patients,features_step)
        x_set[:,:num_components] += means

    # Test the file
    with h5.File(os.path.join(DATA_PATH,'mixture_queryset.hdf5'),'r') as hf:
        print(hf['X'].shape)
        print(hf['X'][:10,:100])


if __name__=='__main__':
    dataset = 'snp'

    if dataset == 'snp':
        generate_snps()
    elif dataset == 'mixture':
        generate_gaussian_mixture()
