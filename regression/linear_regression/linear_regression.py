
import numpy as np

class LinearRegression():

    def fit(self, X, y):
        ones_col = np.ones((X.shape[0]))
        X = np.column_stack((ones_col, X))

        self._coefs = np.linalg.inv(X.T @ X) @ X.T @ y

        return self

    def predict(self, X):
        ones_col = np.ones(X.shape[0])
        X = np.column_stack((ones_col, X))
        
        return X @ self._coefs

