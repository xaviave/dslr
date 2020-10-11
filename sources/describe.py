import logging

from Parser.ArgParser import ArgParser
from Parser.CSVParser import CSVParser
from math import sqrt, floor, ceil


logging.getLogger().setLevel(logging.INFO)


class Describe(CSVParser):
    """
    Take a dataset in parameter and can display some information about it.
    """
    content_vars: dict

    def __init__(self, args: ArgParser):
        super().__init__(args)
        self.desc_vars = {
            "count": len,
            "mean": self._mean,
            "std": self._std_dev,
            "min": self._min,
            "0.25": self._percentile,
            "0.50": self._percentile,
            "0.75": self._percentile,
            "max": self._max,
            "unique": self._unique,
        }
        self.csv_parser()
        self._set_val()

    @staticmethod
    def _percentile(array, percent=0.50):
        # can reduce sorting time by getting already sorted array
        array = array.sort_values(kind="mergesort").values
        try:
            percent = float(percent)
            k = (len(array) - 1) * percent
            f = floor(k)
            c = ceil(k)
            if f == c:
                return array[int(k)]
            d0 = array[int(f)] * (c - k)
            d1 = array[int(c)] * (k - f)
            return d0 + d1
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

    def _get_specific(self, func, *args):
        return {
            feature: func(self.raw_data.get(feature), *args)
            for feature in self.header
        }

    def _set_val(self):
        self.content_vars = {}
        for desc, func in self.desc_vars.items():
            if '.' in desc:  # if it's a float (percentile)
                self.content_vars[desc] = self._get_specific(func, desc)
            else:
                self.content_vars[desc] = self._get_specific(func)

    def _print_header(self):
        print("%6s" % ' ', end='')
        for feature in self.header:
            print("%26s" % feature, end='')
        print('')

    def _print_feature(self):
        for var in self.desc_vars:
            print("%6s" % var, end='')
            for feature in self.header:
                print("%26s" % self.content_vars[var].get(feature), end='')
            print('')

    def describe(self):
        # make a list or parameter to chose feature to print.
        self._print_header()
        self._print_feature()


def main():
    args = ArgParser()
    describer = Describe(args)
    describer.describe()


if __name__ == "__main__":
    main()
