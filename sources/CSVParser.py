import os
import sys
import logging
import argparse

import pandas as pd
import numpy as np


logging.getLogger().setLevel(logging.INFO)


class CSVParser:
    """
    Need to analyse the dataset to highlight the usefull data and create the ANALYZED_HEADER parameter
    All the logisitc Reference is based on this paramter
    """

    header: list
    df: pd.DataFrame
    raw_data: pd.DataFrame
    args: argparse.Namespace

    # not analysed for now
    ANALYZED_HEADER: np.ndarray = [
        "First Name",
        "Last Name",
        "Birthday",
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
            logging.error("CSV file header doesn't contain enought data to analyse the dataset")
            sys.exit(0)

    def _get_csv_file(self):
        logging.info(f"Reading dataset from CSV file: {self.args.file_name}")
        self.raw_data = pd.read_csv(f"{os.path.abspath(self.args.file_name)}")
        self.header = self.raw_data.columns.values
        self._check_header()

    def _as_df(self):
        self.df = pd.DataFrame(data=self.raw_data, columns=self.ANALYZED_HEADER)

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def visualizer(self):
        print(self.df.head())
        # need to create the data visializer with graph and numpy
        pass

    def csv_parser(self):
        self._get_csv_file()
        self._as_df()
