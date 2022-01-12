from matplotlib import pyplot
import numpy as np

class LinearRegression: 
    def __init__(self, X: np.ndarray, y: np.ndarray, ridge_alpha: float = 0) -> None:
        """
        Initialize and fit the model.
        """
        self.ridge_alpha = ridge_alpha
        self.weights = self.__fit(X.T, y)
        print(self.weights)
        

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the target for X.
        """
        # Assume single vector of explanatory variables
        return np.dot(x, self.weights)


    def __fit(self, X: np.ndarray, y = np.ndarray) -> np.ndarray:
        """
        Fit the model with ridge regularization.
        """
        inverse = np.linalg.inv(X.T @ X + self.ridge_alpha * np.identity(X.shape[1]))
        # return np.linalg.inv(X.T @ X + np.eye(X.shape[0]) * (self.ridge_alpha ** 2)) @ X.T @ y
        inv_times_XT = inverse @ X.T
        return inv_times_XT @ y


def main():
    """
    Main function.
    """
    import random
    import matplotlib.pyplot as plt

    x1 = np.linspace(0, 9, 10)
    x2 = np.linspace(2, 20, 10)
    X  = np.array([x1, x2])
    y = np.array([4 * idx + idx_2 + 1 - 2 * random.random() for idx, idx_2 in zip(x1, x2)])
    model = LinearRegression(X, y, ridge_alpha=0.0001)
    pred = np.array([model.predict(X.T[idx]) for idx in range(10)])
    print(pred)
    
    plt.plot(y, y)
    plt.plot(y, pred)
    plt.show()


if __name__ == "__main__":
    main()