# Author: Eljas Roellin
#         Michael Adamer
#         Lucie Bourguignon
#
# License: BSD 3 clause

import numpy as np
from torch.utils.data import DataLoader

# Convert y (for classifier only)
def prepare_classification(y):

    # make y an array
    y = np.array(y)

    # Initialize containers which will be used for predictions
    all_classes = []

    if y.ndim == 1:
        # if y is 1-dimensional: reshape
        y = y.reshape(-1,1)
        y_converted = np.zeros_like(y, dtype=int)
        num_classify = 1
    else:
        y_converted = np.zeros_like(y, dtype=int)
        num_classify = y_converted.shape[1]

    # Create a list of np.arrays "classes"
    # a "classes" array stores the votes per class during prediction
    for k in range(num_classify):
        # Notice that np.unique sorts the elements lexicographically
        classes, y_converted[:, k] = np.unique(y[:, k], return_inverse=True)
        y_converted = y_converted.astype(int)
        all_classes.append(classes)

    return all_classes, y_converted
