
import numpy as np

class LinearRegression():
    def __init__(self):
        self._coefs = None

    def fit(self, X, y):
        ones_col = np.ones((X.shape[0]))
        X = np.column_stack((ones_col, X))

        if(np.linalg.det(X) != 0):
            self._coefs = np.linalg.inv(X.T @ X) @ X.T @ y
        else:
            print("[ WARNING ] Determinant of the data is zero.\n"
                    "Coefficients have not been computed.")

        return self

    def predict(self, X):
        ones_col = np.ones(X.shape[0])
        X = np.column_stack((ones_col, X))
        
        return X @ self._coefs

