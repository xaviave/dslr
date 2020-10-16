import os
import logging
import datetime
import warnings

import numpy as np

from Tools.DatasetHandler import DatasetHandler

# delete the numpy deprecation warning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
logging.getLogger().setLevel(logging.INFO)


class LogReg(DatasetHandler):
    cost: list
    batch: int
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
    def _slice(dataset, y_train, len_batch):
        indices = np.random.permutation(dataset.shape[0])
        return dataset[indices[:len_batch]], y_train[indices[:len_batch]]

    def _sigmoid(self, x):
        try:
            return 1 / (1 + np.exp(-x))
        except Exception as e:
            self._exit(exception=e, message="Error during _sigmoid")

    def _cost_function(self, h, y):
        try:
            m = len(y)
            return (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        except Exception as e:
            self._exit(exception=e, message="Error during _cost_function")

    def _gradient_descent(self, dataset, h, y, m):
        try:
            return self.alpha * np.dot(dataset.T, (h - y)) / m
        except Exception as e:
            self._exit(exception=e, message="Error during _gradient_descent")

    def _batch_gradient_descent(self, dataset, actual_y):
        theta = np.zeros(dataset.shape[1])
        for _ in range(self.n_iter):
            h = self._sigmoid(dataset.dot(theta))
            theta = theta - self._gradient_descent(dataset, h, actual_y, len(dataset))
        return theta

    def _stochastic_gradient_descent(self, dataset, actual_y):
        theta = np.zeros(dataset.shape[1])
        try:
            len_batch = int(len(dataset) / self.batch)
            for _ in range(self.n_iter):
                dataset_train, actual_y_train = self._slice(dataset, actual_y, len_batch)
                h = self._sigmoid(dataset_train.dot(theta))
                theta = theta - self._gradient_descent(
                    dataset_train, h, actual_y_train, len(dataset_train)
                )
        except Exception as e:
            self._exit(exception=e, message="Error during _stochastic_gradient_descent")
        return theta

    def __init__(
        self,
        train: bool = False,
        alpha: float = 0.01,
        batch: int = 35,
        n_iteration: int = 3000,
        theta_file_name: str = "generated_theta.npy",
    ):
        super().__init__(parse=True, train=train)
        self.alpha = alpha
        self.batch = batch
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
        dataset = np.insert(np.nan_to_num(dataset), 0, 1, axis=1)
        for i in np.unique(y):
            actual_y = np.where(y == i, 1, 0)
            theta = self.gradient_func(dataset, actual_y)
            self.theta.append((theta, i))
        logging.info(f"timer={datetime.datetime.now() - start}: Training finish")

    def predict(self, dataset):
        dataset = np.insert(np.nan_to_num(dataset, copy=False), 0, 1, axis=1)
        return [
            max((self._sigmoid(i.dot(theta)), c) for theta, c in self.theta)[1] for i in dataset
        ]

    def score(self, dataset, y):
        try:
            return sum(self.predict(dataset) == y) / len(y)
        except Exception as e:
            self._exit(exception=e, message="Error during _score")
