import os
import sys
import logging

import numpy as np
import pandas as pd

from Tools.ArgParser import ArgParser
from Tools.Visualiser import Visualiser
from Tools.Describer import Describer

logging.getLogger().setLevel(logging.INFO)


class DatasetHandler(ArgParser, Visualiser, Describer):
    header: list
    df: pd.DataFrame
    np_df: np.ndarray
    np_df_train: np.ndarray
    analysed_header: np.array
    default_header_file: str = os.path.join("sources", "Ressources", "analysed_header.npy")

    @staticmethod
    def _load_npy(file_name: str):
        return np.load(file_name, allow_pickle=True)

    @staticmethod
    def _save_npy(file_name: str, data):
        np.save(file_name, data, allow_pickle=True)

    def _check_header(self):
        """
        The logisitc regression is all about data.
        Well choose headers from a list of header in a dataset is the main data analysis part
        this function is oriented for machine learning programs
        """
        self.load_header()
        self.header = list(self.raw_data.columns.values)
        if not all(h in self.header for h in self.analysed_header):
            logging.error("CSV file header doesn't contain enough data to analyse the dataset")
            sys.exit(-1)

    def _get_csv_file(self):
        logging.info(f"Reading dataset from file: {self.argparse_file_name}")
        try:
            self.raw_data = pd.read_csv(f"{os.path.abspath(self.argparse_file_name)}")
            self.raw_data.fillna(0, inplace=True)
        except Exception:
            logging.error(f"Error while processing {self.argparse_file_name}")
            sys.exit(-1)
        self._check_header()

    @staticmethod
    def _normalise(dataset: pd.DataFrame):
        for h in dataset.columns:
            dataset[h] = dataset[h].apply(lambda x: x / max(dataset[h]))
        return dataset.to_numpy()

    def _as_df(self, train: bool, classifier: str = None):
        self.df = pd.DataFrame(data=self.raw_data, columns=self.analysed_header)
        self.np_df = self._normalise(self.df)
        if train and classifier is not None:
            self.np_df_train = self.raw_data[classifier].values

    def __init__(self, parse: bool = False, train: bool = False):
        super().__init__()
        self._get_csv_file()
        if parse:
            self.csv_parser(train=train, classifier=self.get_args("classifier"))
        self.visualize()

    @staticmethod
    def write_to_csv(dataset: list, columns: list):
        tmp_dataset = pd.DataFrame(data=dataset)
        tmp_dataset.index.name = "Index"
        tmp_dataset.columns = columns
        with open(os.path.abspath("houses.csv"), "w") as file:
            file.write(tmp_dataset.to_csv())

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
        """
        adapt this with the parser option
        """
        if self.get_args("visualiser"):
            self.describe(headers=list(self.analysed_header), slice_print=6)
            self.visualizer(self.analysed_header)
