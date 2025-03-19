
import numpy as np

class LinearRegression():
    def __init__(self):
        self._coefs = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if X.ndim != 2:
            X = X.reshape((len(X), 1))
#            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        ones_col = np.ones((X.shape[0]))
        X = np.column_stack((ones_col, X))

        print(X)
        print(y)

        X_squared = X.T @ X
        if np.linalg.det(X_squared) != 0:
            self._coefs = np.linalg.inv(X_squared) @ X.T @ y
        else:
            raise ValueError("Cannot compute the inverse of a singular matrix")
        return self

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim != 2:
            X = X.reshape(len(X), 1)

        ones_col = np.ones(X.shape[0])
        X = np.column_stack((ones_col, X))
        
        return X @ self._coefs

