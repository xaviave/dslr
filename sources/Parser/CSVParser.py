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

    # better if we take this from a file
    ANALYZED_HEADER: np.ndarray = np.array(
        [
            "Best Hand",
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
        if not all(h in self.header for h in self.ANALYZED_HEADER):
            logging.error("CSV file header doesn't contain enough data to analyse the dataset")
            sys.exit(-1)

    def _get_csv_file(self):
        logging.info(f"Reading dataset from CSV file: {self.args.file_name}")
        try:
            self.raw_data = pd.read_csv(
                f"{os.path.abspath(self.args.file_name)}"
            )  # think about error handler parameter
            # dropna is maybe better to don't modify the global value of the dataset
            self.raw_data.fillna(0, inplace=True)
            # self.raw_data.dropna(inplace=True)
        except Exception:
            logging.error(f"Error while processing {self.args.file_name}")
            sys.exit(-1)
        self.header = list(self.raw_data.columns.values)
        self._check_header()

    def _as_df(self):
        self.df = pd.DataFrame(data=self.raw_data, columns=self.ANALYZED_HEADER)
        if "Birthday" in self.df.columns:
            self.df["Birthday"] = self.df["Birthday"].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
            )

    def __init__(self, args: ArgParser):
        self.args = args
        self._get_csv_file()
        super(Visualiser).__init__(pd.DataFrame)
        super(Describer).__init__()

    def csv_parser(self):
        self._as_df()
        if vars(self.args.args).get("visualiser"):
            self.visualizer(self.ANALYZED_HEADER)

    def describe(self, **kwargs):
        # add checker for Describer init
        Describer.describe(data=self.raw_data, headers=kwargs.get("headers", self.header))
