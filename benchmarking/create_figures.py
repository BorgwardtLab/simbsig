import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


DATA_PATH = 'benchmarking'#'/home/michael/ETH/data/SIMBSIG'
n_trials = 10

df_list_cpu = [pd.read_csv(os.path.join(DATA_PATH,f'runtime_df_cpu_{i}.csv'),index_col=0) for i in range(n_trials)]
df_list_gpu = [pd.read_csv(os.path.join(DATA_PATH,f'runtime_df_gpu_{i}.csv'),index_col=0) for i in range(n_trials)]
df_list_ooc = [pd.read_csv(os.path.join(DATA_PATH,f'runtime_df_ooc_{i}.csv'),index_col=0) for i in range(n_trials)]

cols_to_drop = [c for c in df_list_gpu[0].columns if 'cuml' in c]
cols_rename = {c:c+'_gpu' for c in df_list_gpu[0].columns if 'cuml' not in c}
df_list_gpu = [df.drop(cols_to_drop,axis=1).rename(columns=cols_rename) for df in df_list_gpu]

df_cpu_all = pd.concat(df_list_cpu,axis=1)
df_gpu_all = pd.concat(df_list_gpu,axis=1)
df_ooc_all = pd.concat(df_list_ooc,axis=1)

df_figure = pd.concat([df_cpu_all,df_gpu_all,df_ooc_all],axis=1) #.loc[:int(1e5)]
for figure in ['neighbors','kmeans','pca']:
    cols = [c for c in df_figure.columns if figure in c]
    plt.figure(figure)
    sns.lineplot(data=df_figure[cols],legend=False)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    if figure == 'kmeans':
        plt.legend(loc='lower right', labels=['Scikit-learn', 'SIMBSIG CPU', 'SIMBSIG GPU', 'SIMBSIG CPU OOC', 'SIMBSIG GPU OOC'],fontsize=10)
    plt.xlabel('Number of datapoints')
    plt.ylabel('time [s]')

    plt.savefig(os.path.join(DATA_PATH,figure+'.png'))
