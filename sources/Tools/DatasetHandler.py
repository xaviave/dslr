import os
import sys
import logging

import numpy as np
import pandas as pd

from Tools.ArgParser import ArgParser
from Tools.Visualiser import Visualiser
from Tools.Describer import Describer

logging.getLogger().setLevel(logging.INFO)


class DatasetHandler(Visualiser, ArgParser, Describer):
    header: list
    df: pd.DataFrame
    np_df: np.ndarray
    np_df_train: np.ndarray
    analysed_header: np.array
    argparse_file_name: str
    default_dataset: str = os.path.join("data", "datasets", "dataset_train.csv")
    default_header_file: str = os.path.join("sources", "Resources", "analysed_header.npy")

    """
        Override methods
    """

    def _add_parser_args(self, parser):
        super()._add_parser_args(parser)
        parser.add_argument(
            "-f",
            "--csv_file",
            help=f"Provide CSV dataset file - Using '{self.default_dataset}' as default file",
            type=str,
            default=self.default_dataset,
        )

    def _get_options(self):
        self.argparse_file_name = self.args.csv_file
        if self.argparse_file_name == self.default_dataset:
            logging.info("Using default dataset CSV file")
        if (
            not os.path.exists(self.argparse_file_name)
            or os.path.splitext(self.argparse_file_name)[1] != ".csv"
        ):
            self._exit(
                message="The file doesn't exist or is in the wrong format\nProvide a CSV file"
            )

    """
        Private methods
    """

    def _load_npy(self, file_name: str):
        try:
            return np.load(file_name, allow_pickle=True)
        except Exception as e:
            self._exit(exception=e)

    def _save_npy(self, file_name: str, data):
        try:
            np.save(file_name, data, allow_pickle=True)
        except Exception as e:
            self._exit(exception=e)

    def _check_header(self):
        """
        The logistic regression is all about data.
        Well choose headers from a list of header in a dataset is the main data analysis part
        this function is oriented for machine learning programs
        """
        self.load_header()
        self.header = list(self.raw_data.columns.values)
        if not all(h in self.header for h in self.analysed_header):
            self._exit(
                message="CSV file header doesn't contain enough data to analyse the dataset"
            )

    def _get_csv_file(self):
        logging.info(f"Reading dataset from file: {self.argparse_file_name}")
        try:
            self.raw_data = pd.read_csv(f"{os.path.abspath(self.argparse_file_name)}")
            self.raw_data.fillna(0, inplace=True)
        except Exception:
            self._exit(message=f"Error while processing {self.argparse_file_name}")
        self._check_header()

    @staticmethod
    def _normalise(dataset: pd.DataFrame):
        for h in dataset.columns:
            try:
                dataset[h] = dataset[h].apply(lambda x: x / max(dataset[h]))
            except ZeroDivisionError:
                logging.warning(f"0 is max dats from {h} column")
                dataset.drop(h)
        return dataset.to_numpy()

    def _as_df(self, train: bool, classifier: str = None):
        self.df = pd.DataFrame(data=self.raw_data, columns=self.analysed_header)
        self.np_df = self._normalise(self.df)
        if train and classifier is not None:
            try:
                self.np_df_train = self.raw_data[classifier].values
            except KeyError as e:
                self._exit(exception=e, message="Error with getting [classifier] in raw_data")

    def __init__(self, parse: bool = False, train: bool = False):
        super().__init__()
        self._get_csv_file()
        if parse:
            self.csv_parser(train=train, classifier=self.get_args("classifier"))
        self.visualize()

    """
        Public methods
    """

    def write_to_csv(self, dataset: list, columns: list):
        tmp_dataset = pd.DataFrame(data=dataset)
        tmp_dataset.index.name = "Index"
        tmp_dataset.columns = columns
        try:
            with open(os.path.abspath("houses.csv"), "w") as file:
                file.write(tmp_dataset.to_csv())
        except Exception as e:
            self._exit(exception=e)

    def save_header(self, header_file: str = default_header_file):
        self._save_npy(header_file, self.analysed_header)

    def load_header(self, header_file: str = default_header_file):
        self.analysed_header = self._load_npy(header_file)

    def describe(self, **kwargs):
        Describer.describe(
            data=self.raw_data,
            headers=kwargs.get("headers", self.header),
            slice_print=kwargs.get("slice_print", 4),
        )

    def csv_parser(self, train: bool = False, classifier: str = None):
        self._as_df(train, classifier=classifier)

    def visualize(self):
        if self.get_args("type_visualizer") is not None:
            self.describe(headers=list(self.analysed_header), slice_print=4)
            self.visualizer(self.analysed_header)
