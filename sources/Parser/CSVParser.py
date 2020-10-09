import os
import sys
import logging
import datetime

import numpy as np
import pandas as pd

from Parser.ArgParser import ArgParser
from Visualiser.Visualiser import Visualiser

logging.getLogger().setLevel(logging.INFO)


class CSVParser(Visualiser):
    """
    Need to analyse the dataset to highlight the useful data and create the ANALYZED_HEADER parameter
    All the logistic Reference is based on this parameter
    """

    header: list
    args: ArgParser
    df: pd.DataFrame
    # not analysed for now
    # better if we take this from a file
    ANALYZED_HEADER: np.ndarray = [
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

    def _check_header(self):
        if not all(h in self.header for h in self.ANALYZED_HEADER):
            logging.error(
                "CSV file header doesn't contain enough data to analyse the dataset"
            )
            sys.exit(-1)

    def _get_csv_file(self):
        logging.info(f"Reading dataset from CSV file: {self.args.file_name}")
        try:
            self.raw_data = pd.read_csv(
                f"{os.path.abspath(self.args.file_name)}"
            )  # think about error handler parameter
        except Exception:
            logging.error(f"Error while processing {self.args.file_name}")
            sys.exit(-1)
        self.header = self.raw_data.columns.values
        self._check_header()

    def _as_df(self):
        self.df = pd.DataFrame(data=self.raw_data, columns=self.ANALYZED_HEADER)
        if "Birthday" in self.df.columns:
            self.df["Birthday"] = self.df["Birthday"].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
            )

    def __init__(self, args: ArgParser):
        self.args = args

    def csv_parser(self):
        self._get_csv_file()
        self._as_df()
        if "visual" in self.args.options.keys():
            self.visualizer(self.ANALYZED_HEADER)
