
import numpy as np

class LinearRegression():
    """
    A simple linear regression model implemented using NumPy.

    ...

    Attributes
    ----------
    coefs_ : numpy.ndarray
        The coefficients of the linear regression model (including the intercept).
        
        The first element is the intercept, and the subsequent elements correspond
        to the coefficients for the input features.
    
    Methods
    -------
    fit(X, y)
        Computes the coefficients to fit the provided data.
    predict(X)
        Returns the predicted values of the given values.
    fit_predict(X, y)
        First, fits the model, and then returns the predictions of the training data.
    get_intercept()
        Returns the intercept of the linear model.
    get_coefs()
        Returns the coefficients of the linear model.

    """

    def __init__(self):
        """
        Initializes the LinearRegression model.
        """
        self.coefs_ = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if X.ndim != 2:
            X = X.reshape((len(X), 1))
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        ones_col = np.ones((X.shape[0]))
        X = np.column_stack((ones_col, X))

        X_squared = X.T @ X
        if np.linalg.det(X_squared) != 0:
            self.coefs_ = np.linalg.inv(X_squared) @ X.T @ y
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
        
        return X @ self.coefs_

    def fit_predict(self, X, y):
        self = self.fit(X, y)
        return self.predict(X)

    def get_intercept(self):
        return self.coefs_[1]

    def get_coefs(self):
        return self.coefs_[1:]

