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

#Seaborn

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
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax = sns.lineplot(data=df_figure[cols],ax=ax1,legend=False)# if figure != 'pca' else 'auto')
    # ax = sns.lineplot(data=df_figure[cols],ax=ax2,legend=False)
    #
    # if figure == 'pca':
    #     ax1.legend(loc='upper left', labels=['Scikit-learn', 'SIMBSIG CPU', 'SIMBSIG GPU', 'SIMBSIG CPU OOC', 'SIMBSIG GPU OOC'],fontsize=10)
    #
    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.yaxis.set_visible(False)
    #
    # ax1.set_xlim(0, 100000)
    # ax2.set_xlim(400000, 500000)
    #
    # ax1.set_xticks(np.arange(0,100001,20000))
    # ax2.set_xticks(np.arange(400000,500001,20000))
    #
    # ax1.set_xticklabels(np.arange(0,5e5+1,2e4)/1e5)
    # ax2.ticklabel_format(axis='x',style='sci', scilimits=(0,0))
    #
    # d = .015 # how big to make the diagonal lines in axes coordinates
    # # arguments to pass plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((1-d,1+d), (-d,+d), **kwargs)
    # ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)
    #
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    # ax2.plot((-d,+d), (-d,+d), **kwargs)
    #
    # fig.text(0.5, 0.02, 'Number of datapoints', ha='center')
    # fig.text(0.04, 0.5, 'time [s]', va='center', rotation='vertical')
    plt.savefig(os.path.join(DATA_PATH,figure+'.png'))



# Pyplot only

# cols_cpu_final = np.array([[c+'_mean', c+'_std'] for c in df_list_cpu[0].columns]).flatten()
# df_cpu_final = pd.DataFrame(index=df_list_cpu[0].index,columns=cols_cpu_final)
#
# cols_gpu_final = np.array([[c+'_mean', c+'_std'] for c in df_list_gpu[0].columns]).flatten()
# df_gpu_final = pd.DataFrame(index=df_list_gpu[0].index,columns=cols_gpu_final)
#
# cols_ooc_final = np.array([[c+'_mean', c+'_std'] for c in df_list_ooc[0].columns]).flatten()
# df_ooc_final = pd.DataFrame(index=df_list_ooc[0].index,columns=cols_ooc_final)
# for data in ['cpu','gpu','ooc']:
#     df_final = eval(f'df_{data}_final')
#     df_all = eval(f'df_{data}_all')
#     cols = eval(f'df_list_{data}[0].columns')
#
#     for c in cols:
#         df_final.loc[:,[c+'_mean', c+'_std']] = pd.concat([df_all[c].mean(axis=1),df_all[c].std(axis=1)],axis=1).values
#
#
# df_figure = pd.concat([df_cpu_final,df_gpu_final,df_ooc_final.loc[:int(1e5)]],axis=1)
#
# for figure in ['neighbors','kmeans','pca']:
#     cols = [c for c in df_figure.columns if figure in c]
#     cols_means = [c for c in cols if '_mean' in c]
#     cols_stds = [c for c in cols if '_std' in c]
#
#     fig = plt.figure(figure)
#
#     for i in range(len(cols_means)):
#         label = cols_means[i].replace(figure+'_','').replace('_mean','')
#         plt.errorbar(df_figure[cols_means[i]].index.values,
#                      df_figure[cols_means[i]].values,
#                      yerr=df_figure[cols_stds[i]].values,
#                      label=label)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
