import os
import sys
import logging
import argparse


logging.getLogger().setLevel(logging.INFO)


class ArgParser:
    options: dict
    file_name: str
    args: argparse.Namespace
    default: str = os.path.join("data", "datasets", "dataset_train.csv")

    def __init__(self):
        self.data = {}
        self._init_argparse()
        self._get_options()

    def _init_argparse(self):
        """
        custom arguments to add option
        """
        parser = argparse.ArgumentParser(description="Process CSV dataset")
        parser.add_argument(
            "-f",
            "--csv_file",
            help=f"Provide CSV dataset file - Using '{self.default}' as default file",
            type=str,
            default=self.default,
        )
        parser.add_argument("-p", action="store_true", help="Render the progression")
        self.args = parser.parse_args()

    def _get_options(self):
        self.file_name = self.args.csv_file
        self.options = {"progress": self.args.p}
        if self.file_name == self.default:
            logging.info("Using default dataset CSV file")
        if not os.path.exists(self.file_name) or os.path.splitext(self.file_name)[1] != ".csv":
            logging.error("The file doesn't exist or is in the wrong format.\nProvide a CSV file")
            sys.exit(-1)
