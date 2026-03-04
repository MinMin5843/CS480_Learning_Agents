import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(batch_size=32):
    """
    Splits data from the input and target files into training, validation, and test sets.

    Args:
        batch_size: the number of samples per batch during training that defaults to 32. 

    Yields:
        A tuple containing the training, validation, and test data sets. 
    """
    X = np.load("data/inputs.npy").astype("float32")
    y = np.load("data/targets.npy").astype("int64")

    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val),   torch.tensor(y_val)),   batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(torch.tensor(X_test),  torch.tensor(y_test)),  batch_size=batch_size)

    return train_loader, val_loader, test_loader