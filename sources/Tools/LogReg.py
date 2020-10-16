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

    """
        Override methods
    """

    def _add_parser_args(self, parser):
        super()._add_parser_args(parser)
        parser.add_argument(
            "-c",
            "--classifier",
            help=f"Provide a classifier name from the dataset column",
            type=str,
        )

    def _add_exclusive_args(self, parser):
        super()._add_exclusive_args(parser)
        gradient_group = parser.add_mutually_exclusive_group(required=False)
        gradient_group.add_argument(
            "-bgd",
            "--batch_gradient_descent",
            action="store_const",
            const=self._batch_gradient_descent,
            help="Use batch gradient descent as optimisation algorithm (default value)",
            dest="type_gradient",
        )
        gradient_group.add_argument(
            "-sgd",
            "--stochastic_gradient_descent",
            action="store_const",
            const=self._stochastic_gradient_descent,
            help="Use stochastic gradient descent as optimisation algorithm",
            dest="type_gradient",
        )

    def _get_options(self):
        super()._get_options()
        if self.get_args("classifier") is None:
            self._exit(message="classifier option not provided")

    """
        Private methods
    """

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _cost_function(h, y):
        m = len(y)
        return (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))

    def _gradient_descent(self, dataset, h, y, m):
        return self.alpha * (np.dot(dataset.T, (h - y)) / m)

    def _batch_gradient_descent(self, dataset, theta, actual_y, y):
        for _ in range(self.n_iter):
            h = self._sigmoid(dataset.dot(theta))
            theta -= self._gradient_descent(dataset, h, actual_y, len(y))
        return theta

    def _stochastic_gradient_descent(self, dataset, theta, actual_y, y):
        self._exit("Not implemented")

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
        self.gradient_func = self.get_args(
            "type_gradient", default_value=self._batch_gradient_descent
        )

    """
        Public methods
    """

    def load_theta(self):
        logging.info(f"{datetime.datetime.now()}: Loading '{self.theta_file_name}'")
        self.theta = self._load_npy(self.theta_file_name)

    def save_theta(self):
        logging.info(f"{datetime.datetime.now()}: Saving data into '{self.theta_file_name}'")
        self._save_npy(self.theta_file_name, self.theta)

    def train(self, dataset, y):
        self.theta = []
        start = datetime.datetime.now()
        logging.info("Fitting the given dataset.")
        dataset = np.insert(np.nan_to_num(dataset, copy=False), 0, 1, axis=1)
        for i in np.unique(y):
            actual_y = np.where(y == i, 1, 0)
            theta = self.gradient_func(dataset, np.zeros(dataset.shape[1]), actual_y, y)
            self.theta.append((theta, i))
        logging.info(f"timer={datetime.datetime.now() - start}: Training finish")

    def predict(self, dataset):
        dataset = np.insert(np.nan_to_num(dataset, copy=False), 0, 1, axis=1)
        return [
            max((self._sigmoid(i.dot(theta)), c) for theta, c in self.theta)[1] for i in dataset
        ]

    def score(self, dataset, y):
        return sum(self.predict(dataset) == y) / len(y)
