import numpy as np


def mse(y_true, y_pred):
    np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
    return np.mean(np.square(y_pred - y_true))


def mse_d(y_true, y_pred):
    np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
    return np.mean(2 * (y_pred - y_true) / np.size(y_true), axis=0)

def cross_entropy(y_true, y_pred):
    np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_d(y_true, y_pred):
    np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
    return np.mean( y_pred - y_true, axis=0)
    


def binary_cross_entropy(y_true, y_pred):
    np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_d(y_true, y_pred):
    np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
    return np.mean(((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true), axis=0)
