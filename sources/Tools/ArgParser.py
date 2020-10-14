import logging
import argparse

logging.getLogger().setLevel(logging.INFO)


class ArgParser:
    args: argparse.Namespace

    """
        Override methods
    """

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

    """
        Private methods
    """

    def _add_parser_args(self, parser):
        parser.add_argument("-h", "--help", action=self._HelpAction, help="help usage")

    @staticmethod
    def _add_exclusive_args(parser):
        # @ define it in your class
        pass

    @staticmethod
    def _add_subparser_args(parser):
        # @ define it in your class
        pass

    def _get_options(self):
        # @ define it in your class
        pass

    def _init_argparse(self):
        """
        custom arguments to add option
        """
        self.parser = argparse.ArgumentParser(prog="PROG", add_help=False)
        self._add_parser_args(self.parser)
        self._add_exclusive_args(self.parser)
        self._add_subparser_args(self.parser)
        self.args = self.parser.parse_args()

    def __init__(self):
        self._init_argparse()
        self._get_options()

    """
        Public methods
    """

    def get_args(self, value: str, default=None):
        """
        Allow to access directly and safely to a variable presentor not in the class
        """
        return vars(self.args).get(value, default)
