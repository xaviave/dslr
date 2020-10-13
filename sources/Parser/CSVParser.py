import os
import sys
import logging
import datetime

import numpy as np
import pandas as pd

from Parser.ArgParser import ArgParser
from Visualiser.Visualiser import Visualiser
from Describer.Describer import Describer

logging.getLogger().setLevel(logging.INFO)


class CSVParser(Visualiser, Describer):
    """
    Need to analyse the dataset to highlight the useful data and create the ANALYZED_HEADER parameter
    All the logistic Reference is based on this parameter
    """

    header: list
    args: ArgParser
    df: pd.DataFrame
    np_df: np.ndarray
    np_df_train: np.ndarray

    # better if we take this from a file
    ANALYZED_HEADER: np.ndarray = np.array(
        [
            # "Best Hand",
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Care of Magical Creatures",
            "Charms",
            "Flying",
        ]
    )

    def _check_header(self):
        self.header = list(self.raw_data.columns.values)
        if not all(h in self.header for h in self.ANALYZED_HEADER):
            logging.error("CSV file header doesn't contain enough data to analyse the dataset")
            sys.exit(-1)

    def _get_csv_file(self):
        logging.info(f"Reading dataset from file: {self.args.file_name}")
        try:
            self.raw_data = pd.read_csv(f"{os.path.abspath(self.args.file_name)}")
            self.raw_data.fillna(0, inplace=True)
        except Exception:
            logging.error(f"Error while processing {self.args.file_name}")
            sys.exit(-1)
        self._check_header()

    @staticmethod
    def _normalise(dataset: pd.DataFrame):
        for h in dataset.columns:
            dataset[h] = dataset[h].apply(lambda x: x / max(dataset[h]))
        return dataset.to_numpy()

    def _as_df(self, train: bool):
        # if "Best Hand" in self.raw_data.columns:
        #     self.raw_data["Best Hand"] = self.raw_data["Best Hand"].apply(
        #         lambda x: 0 if x == "Left" else 1
        #     )
        self.df = pd.DataFrame(data=self.raw_data, columns=self.ANALYZED_HEADER)
        self.np_df = self._normalise(self.df)
        if train:
            self.np_df_train = self.raw_data["Hogwarts House"].values

    def __init__(self, args: ArgParser, parse: bool = False, train: bool = True):
        self.args = args
        self._get_csv_file()
        super(Visualiser).__init__(pd.DataFrame)
        super(Describer).__init__()
        if parse:
            self.csv_parser(train)

    def csv_parser(self, train: bool):
        self._as_df(train)
        if vars(self.args.args).get("visualiser"):
            self.visualizer(self.ANALYZED_HEADER)

    def describe(self, **kwargs):
        # add checker for Describer init
        Describer.describe(data=self.raw_data, headers=kwargs.get("headers", self.header))

    @staticmethod
    def write_to_csv(dataset: list, columns: tuple = "Hogwarts House"):
        tmp_dataset = pd.DataFrame(data=dataset)
        tmp_dataset.index.name = "Index"
        tmp_dataset.columns = [columns]
        with open(os.path.abspath("houses.csv"), "w") as file:
            file.write(tmp_dataset.to_csv())
