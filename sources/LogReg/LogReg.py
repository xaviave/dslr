import os
import logging
import datetime
import warnings

import numpy as np

# delete the numpy deprecation warning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
logging.getLogger().setLevel(logging.INFO)


class LogReg:
    cost: list
    theta: list
    n_iter: int
    alpha: float

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _cost_function(h, y):
        m = len(y)
        return (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))

    def _gradient_descent(self, dataset, h, theta, y, m):
        return theta - self.alpha * (np.dot(dataset.T, (h - y)) / m)

    def __init__(self, alpha=0.01, n_iteration=3000, file_name="generated_theta.npy"):
        self.alpha = alpha  # value in the object
        self.n_iter = n_iteration
        self.file_name = file_name

    def load_theta(self):
        logging.info(f"{datetime.datetime.now()}: Loading '{self.file_name}'")
        self.theta = np.load(os.path.abspath(self.file_name), allow_pickle=True)

    def save_theta(self):
        logging.info(f"{datetime.datetime.now()}: Saving data into '{self.file_name}'")
        np.save(self.file_name, self.theta, allow_pickle=True)

    def train(self, dataset, y):
        self.cost = []
        self.theta = []
        start = datetime.datetime.now()
        logging.info("Fitting the given dataset.")
        dataset = np.insert(np.nan_to_num(dataset), 0, 1, axis=1)
        for i in np.unique(y):
            cost = []
            actual_y = np.where(y == i, 1, 0)
            theta = np.zeros(dataset.shape[1])
            for _ in range(self.n_iter):
                h = self._sigmoid(dataset.dot(theta))
                theta = self._gradient_descent(dataset, h, theta, actual_y, len(y))
                cost.append(self._cost_function(h, actual_y))
            self.theta.append((theta, i))
            self.cost.append((cost, i))
        logging.info(f"timer={datetime.datetime.now() - start}: Training finish")

    def predict(self, dataset):
        dataset = np.insert(np.nan_to_num(dataset), 0, 1, axis=1)
        return [
            max((self._sigmoid(i.dot(theta)), c) for theta, c in self.theta)[1] for i in dataset
        ]

    def score(self, dataset, y):
        return sum(self.predict(dataset) == y) / len(y)
