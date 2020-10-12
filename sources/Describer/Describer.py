import logging

from Parser.ArgParser import ArgParser
from Parser.CSVParser import CSVParser
from math import sqrt, floor, ceil
from copy import deepcopy


logging.getLogger().setLevel(logging.INFO)


class Describer(CSVParser):
    """
    Take a dataset in parameter and can display some information about it.
    """
    content_vars: dict

    def __init__(self, args: ArgParser):
        super().__init__(args)
        self.desc_vars = {
            "count": {
                'func': len,
                'param': {}
            },
            "mean": {
                'func': self._mean,
                'param': {}
            },
            "std": {
                'func': self._std_dev,
                'param': {}
            },
            "min": {
                'func': self._min,
                'param': {}
            },
            "25%": {
                'func': self._percentile,
                'param': {'value': 0.25}
            },
            "50%": {
                'func': self._percentile,
                'param': {'value': 0.50}
            },
            "75%": {
                'func': self._percentile,
                'param': {'value': 0.75}
            },
            "max": {
                'func': self._max,
                'param': {}
            },
            "unique": {
                'func': self._unique,
                'param': {}
            },
        }
        self.csv_parser()
        self._set_val()

    @staticmethod
    def _percentile(array, **kwargs):
        # can reduce sorting time by getting already sorted array
        array = array.sort_values(kind="mergesort").values
        try:
            k = (len(array) - 1) * kwargs.get('value')
            f = floor(k)
            c = ceil(k)
            if f == c:
                return array[int(k)]
            return (array[int(f)] * (c - k)) + (array[int(c)] * (k - f))
        except (ValueError, TypeError):
            return "NaN"

    @staticmethod
    def _unique(array):
        unique = {}
        for value in array:
            if value not in unique:
                unique[value] = True
        return len(unique)

    @staticmethod
    def _max(array):
        maximum = array[0]
        for value in array:
            if value > maximum:
                maximum = value
        return maximum

    @staticmethod
    def _min(array):
        minimum = array[0]
        for value in array:
            if value < minimum:
                minimum = value
        return minimum

    @staticmethod
    def _variance(data):
        n = len(data)
        try:
            mean = sum(data) / n
        except (TypeError, ZeroDivisionError):
            return "NaN"
        return sum((x - mean) ** 2 for x in data) / n

    def _std_dev(self, array):
        try:
            return sqrt(self._variance(array))
        except TypeError:
            return "NaN"

    @staticmethod
    def _mean(array):
        try:
            return sum(array) / len(array)
        except (TypeError, ZeroDivisionError):
            return "NaN"

    def _get_specific(self, func):
        return {
            feature: func.get('func')(
                self.raw_data.get(feature),
                **func.get('param')
            )
            for feature in self.header
        }

    def _set_val(self):
        self.content_vars = {
            desc: self._get_specific(func)
            for desc, func in self.desc_vars.items()
        }

    @staticmethod
    def _print_header(headers, to_print):
        print(f"{' ':>6}", end='')
        for feature in headers:
            print(f"{feature if not to_print else '_' * len(feature):>32}", end='')
        print()

    @staticmethod
    def _print_desc(headers, desc_vars, content_vars):
        for var in desc_vars:
            print(f"{var:>6}", end='')
            for feature in headers:
                try:
                    print(f"{content_vars[var].get(feature):>32.6f}", end='')  # for float
                except ValueError:
                    print(f"{content_vars[var].get(feature):>32}", end='')
            print()  # print newline

    def describe(self):
        # make a list or parameter to chose feature to print.
        print(f"{self.__class__.__name__}:")
        copy_header = deepcopy(self.header)
        while copy_header:  # to get only 5 header per 5 to print
            tmp_header = [
                copy_header.pop(0)
                for _ in range(4)
                if len(copy_header)
                          ]
            for to_print in range(2):
                self._print_header(tmp_header, bool(to_print))
            self._print_desc(tmp_header, self.desc_vars, self.content_vars)
            if copy_header:
                print('\n\n')
