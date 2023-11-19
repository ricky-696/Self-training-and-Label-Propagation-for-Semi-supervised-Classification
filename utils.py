import os
import torch
import random
import pickle
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def select_balance_data(dataset, train_idx, n_test=70):
    labels = np.array(dataset.target)[train_idx]
    classes = np.unique(labels)

    ixs = []
    for cl in classes:
        ixs.append(np.random.choice(np.nonzero(labels==cl)[0], n_test,
                replace=False))

    # take same num of samples from all classes
    # ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    # ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])
    ix_unlabel = train_idx[ixs]
    ix_label = train_idx[not ixs]

    return ix_label, ix_unlabel


def save_object(obj, filename):
    try:
        with open(f"{filename}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(f"{filename}.pickle", "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)