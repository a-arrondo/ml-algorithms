
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
    fit(X, y, l2_penalty=0)
        Computes the coefficients to fit the provided data.
    predict(X)
        Returns the predicted values of the given values.
    fit_predict(X, y, l2_penalty=0)
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
        self.l2_penalty_ = 0

    def fit(self, X, y, l2_penalty=0):
        """
        Computes the coefficients to fit the provided data.

        The parameter 'l2_penalty' may be adjusted to apply ridge regression.
        By default, it is disabled: l2_penalty = 0.

        Returns
        -------
        The fitted linear model object.

        Raises
        ------
        ValueError
            If parameter y is 1D, or if the inverse cannot be computed
            due to a singular matrix.
        """
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

        self.l2_penalty_ = l2_penalty
        X_squared = X.T @ X + l2_penalty * np.identity(X.shape[1]) 
        if np.linalg.det(X_squared) != 0:
            self.coefs_ = np.linalg.inv(X_squared) @ X.T @ y
        else:
            raise ValueError("Cannot compute the inverse of a singular matrix")
        return self

    def predict(self, X):
        """
        Calculates the predictions of the given input values.

        Returns
        -------
        A 1D NumPy array with the predictions.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim != 2:
            X = X.reshape(len(X), 1))

        ones_col = np.ones(X.shape[0])
        X = np.column_stack((ones_col, X))
        
        return X @ self.coefs_

    def fit_predict(self, X, y, l2_penalty=0):
        """
        First, fits a model, and then predicts the training values.

        The parameter 'l2_penalty' may be adjusted to apply ridge regression.
        By default, it is disabled: l2_penalty = 0.

        Raises
        ------
        ValueError
            If parameter y is 1D, or if the inverse cannot be computed
            due to a singular matrix.

        Returns
        -------
        A 1D NumPy array with the predictions.
        """
        self = self.fit(X, y, l2_penalty)
        return self.predict(X)

    def get_intercept(self):
        """
        Returns the intercept of the linear model.
        """
        return self.coefs_[1]

    def get_coefs(self):
        """
        Returns the coefficients of the linear model.
        """
        return self.coefs_[1:]

