from matplotlib import pyplot
import numpy as np


class LinearRegression:
    def __init__(self, X: np.ndarray, y: np.ndarray, ridge_alpha: float = 0) -> None:
        """
        Initialize and fit the model.
        """
        self.ridge_alpha = ridge_alpha
        self.weights = self.__fit(X.T, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the target for X.
        """
        # Assume single vector of explanatory variables
        return np.dot(x, self.weights)

    def __fit(self, X: np.ndarray, y=np.ndarray) -> np.ndarray:
        """
        Fit the model with ridge regularization.
        """
        return np.linalg.inv(X @ X.T + self.ridge_alpha * np.identity(X.shape[0])) @ X @ y


if __name__ == "__main__":
    """
    simple function for testing the model.
    """
    import random
    import matplotlib.pyplot as plt

    x1 = np.linspace(0, 9, 10)
    x2 = np.linspace(2, 20, 10)
    X = np.array([x1, x2]).T
    y = np.array([[4 * idx + idx_2 + 0.25 - 0.5 * random.random(), -idx + 2*idx_2] for idx, idx_2 in zip(x1, x2)])

    model = LinearRegression(X, y, ridge_alpha=0.005)
    pred = np.array([model.predict([idx, idx]) for idx in range(10)])
    print(pred)

    plt.plot([i for i in range(10)], pred)
    plt.show()
