import numpy as np
from numpy import linalg as LA

class Supervised_PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit_transform(self, X, y):
        X = np.transpose(X)
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n = X.shape[1]
        H = np.eye(n) - ((1/n) * np.ones((n,n)))
        B = np.matmul(np.transpose(y), y)
        eig_val, eig_vec = LA.eigh( X.dot(H).dot(B).dot(H).dot(np.transpose(X)) )
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            U = eig_vec[:, :self.n_components]
        else:
            U = eig_vec
        X_transformed = np.transpose(U).dot(X)
        X_transformed = np.transpose(X_transformed)
        return X_transformed
