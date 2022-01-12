import numpy as np

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression


def main():
    d = VoiceData()
    model = LinearRegression(X, y, ridge_alpha=0.005)
    pred = np.array([model.predict(X.T[idx]) for idx in range(10)])


if __name__ == '__main__':
    main()
