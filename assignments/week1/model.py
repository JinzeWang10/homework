import numpy as np


class LinearRegression:
    """
    A linear regression model that uses closed form solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.ndarray(10)
        self.b = 0
        # raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        update w with closed form solution.
        """
        # print(X.shape)
        # print(y.shape)
        add_one = np.ones((len(X), 1))
        X = np.append(add_one, X, axis=1)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        # raise NotImplementedError()
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict using updated w
        """
        add_one = np.ones((len(X), 1))
        X = np.append(add_one, X, axis=1)
        return X @ self.w


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        update w using gradient descent
        """
        # self.w=np.linalg.inv(X) @ y

        # print(self.w)
        # print(X)
        # print(y)
        # print(X)
        # print(y)
        add_one = np.ones((len(X), 1))
        X = np.append(add_one, X, axis=1)
        # print(X)
        try:
            self.w = np.zeros(shape=(X.shape[1], y.shape[1]))
        except:
            self.w = np.zeros(shape=(X.shape[1],))
        # print(self.w)
        for i in range(epochs):
            grad = X.T @ X @ (self.w) - X.T @ y
            self.w -= lr * grad
            # print(self.w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        add_one = np.ones((len(X), 1))
        X = np.append(add_one, X, axis=1)
        # print(self.w)
        return X @ self.w
