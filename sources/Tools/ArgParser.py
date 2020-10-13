import os
import sys
import logging
import argparse

logging.getLogger().setLevel(logging.INFO)


class ArgParser:
    argparse_file_name: str
    args: argparse.Namespace
    default_dataset: str = os.path.join("data", "datasets", "dataset_train.csv")

    class _HelpAction(argparse._HelpAction):
        def __call__(self, parser, namespace, values, option_string=None):
            parser.print_help()
            subparsers_actions = [
                action
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
            ]
            for subparsers_action in subparsers_actions:
                for choice, subparser in subparsers_action.choices.items():
                    print(f"Subparser '{choice}'\n{subparser.format_help()}")
            parser.exit()

    def _add_parser_args(self, parser):
        parser.add_argument("-h", "--help", action=self._HelpAction, help="help usage")
        parser.add_argument(
            "-f",
            "--csv_file",
            help=f"Provide CSV dataset file - Using '{self.default_dataset}' as default file",
            type=str,
            default=self.default_dataset,
        )

    @staticmethod
    def _add_exclusive_args(parser):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "-v",
            "--visualiser",
            action="store_const",
            const="v",
            help="Render a tab to vizualize data",
            dest="type_visualizer",
        )
        group.add_argument(
            "-hh",
            "--histogram",
            action="store_const",
            const="hh",
            help="Render an histogram for a specific data",
            dest="type_visualizer",
        )
        group.add_argument(
            "-sc",
            "--scatter_plot",
            action="store_const",
            const="sc",
            help="Render a scatter plot graph for a specific data",
            dest="type_visualizer",
        )
        group.add_argument(
            "-pp",
            "--pair_plot",
            action="store_const",
            const="pp",
            help="Render a pair plot graph for a specific data",
            dest="type_visualizer",
        )

    def _init_argparse(self):
        """
        custom arguments to add option
        """
        self.parser = argparse.ArgumentParser(
            prog="PROG", add_help=False, description="Process CSV dataset"
        )
        self._add_parser_args(self.parser)
        self._add_exclusive_args(self.parser)
        self.args = self.parser.parse_args()

    def _get_options(self):
        self.argparse_file_name = self.args.csv_file
        if self.argparse_file_name == self.default_dataset:
            logging.info("Using default dataset CSV file")
        if (
            not os.path.exists(self.argparse_file_name)
            or os.path.splitext(self.argparse_file_name)[1] != ".csv"
        ):
            logging.error("The file doesn't exist or is in the wrong format\nProvide a CSV file")
            sys.exit(-1)

    def __init__(self):
        self._init_argparse()
        self._get_options()

    def get_args(self, value: str, default=None):
        return vars(self.args).get(value, default)
