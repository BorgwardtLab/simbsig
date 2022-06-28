# Author: Michael Adamer
#         Eljas Roellin
#         Lucie Bourguignon
#
# License: BSD 3 clause

import numpy as np
from torch.utils.data import Dataset

class arraysDataset(Dataset):
    def __init__(self,X,**kwargs):#,y=None):
        super().__init__()
        self.X = np.array(X)

    def __getitem__(self,idx):
        return self.X[idx].astype(np.float32)

    def n_features(self):
        return self.X.shape[1]

    def __len__(self):
        return len(self.X)


class hdf5Dataset(Dataset):
    def __init__(self,data,**kwargs):
        super().__init__()
        self.X = data[kwargs.pop('X_path', 'X')]
        try:
            self.y = data[kwargs.pop('y_path', 'y')]
        except KeyError:
            pass

    def __getitem__(self,idx):
        return self.X[idx].astype(np.float32)#, self.y[idx]]

    def n_features(self):

        return self.X.shape[1]

    def __len__(self):
        return len(self.X)
