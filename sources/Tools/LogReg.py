import os
import logging
import datetime
import sys
import warnings

import numpy as np

from Tools.DatasetHandler import DatasetHandler

# delete the numpy deprecation warning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
logging.getLogger().setLevel(logging.INFO)


class LogReg(DatasetHandler):
    cost: list
    theta: list
    n_iter: int
    alpha: float
    theta_file_name: str

    def _add_parser_args(self, parser):
        super()._add_parser_args(parser)
        parser.add_argument(
            "-c",
            "--classifier",
            help=f"Provide a classifier name from the dataset column",
            type=str,
            default="Hogwarts House",
        )

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _cost_function(h, y):
        m = len(y)
        return (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))

    def _gradient_descent(self, dataset, h, theta, y, m):
        return theta - self.alpha * (np.dot(dataset.T, (h - y)) / m)

    def __init__(
        self,
        train: bool = False,
        alpha: float = 0.01,
        n_iteration: int = 3000,
        theta_file_name: str = "generated_theta.npy",
    ):
        super().__init__(parse=True, train=train)
        self.alpha = alpha
        self.n_iter = n_iteration
        self.theta_file_name = os.path.abspath(theta_file_name)

    def load_theta(self):
        logging.info(f"{datetime.datetime.now()}: Loading '{self.theta_file_name}'")
        self.theta = self._load_npy(self.theta_file_name)

    def save_theta(self):
        logging.info(f"{datetime.datetime.now()}: Saving data into '{self.theta_file_name}'")
        self._save_npy(self.theta_file_name, self.theta)

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
                # look how to optimise number of iteration with actual and previous cost
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
