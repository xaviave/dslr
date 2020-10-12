import itertools
import re

import pandas as pd
import logging

from math import sqrt, floor, ceil

logging.getLogger().setLevel(logging.INFO)


class Describer:
    """
    Take a dataset in parameter and can display some information about it.
    """

    content_vars: dict = {}

    @staticmethod
    def _get_percentile(percentile):
        p = re.match(r"^(?P<perc>\d{2})%$", percentile).group("perc")
        return float(int(p) / 100)

    @classmethod
    def _percentile(cls, array, **kwargs):
        # can reduce sorting time by getting already sorted array
        array = array.sort_values(kind="mergesort").values
        try:
            k = (len(array) - 1) * cls._get_percentile(kwargs.get("value"))
            f = floor(k)
            c = ceil(k)
            if f == c:
                return array[int(k)]
            return (array[int(f)] * (c - k)) + (array[int(c)] * (k - f))
        except (ValueError, TypeError):
            return "NaN"

    @staticmethod
    def _unique(array, **kwargs):
        unique = {}
        for value in array:
            if value not in unique:
                unique[value] = True
        return len(unique)

    @staticmethod
    def _max(array, **kwargs):
        maximum = array[0]
        for value in array:
            if value > maximum:
                maximum = value
        return maximum

    @staticmethod
    def _min(array, **kwargs):
        minimum = array[0]
        for value in array:
            if value < minimum:
                minimum = value
        return minimum

    @staticmethod
    def _variance(data, **kwargs):
        n = len(data)
        try:
            mean = sum(data) / n
        except (TypeError, ZeroDivisionError):
            return "NaN"
        return sum((x - mean) ** 2 for x in data) / n

    @classmethod
    def _std_dev(cls, array, **kwargs):
        try:
            return sqrt(cls._variance(array))
        except TypeError:
            return "NaN"

    @staticmethod
    def _mean(array, **kwargs):
        try:
            return sum(array) / len(array)
        except (TypeError, ZeroDivisionError):
            return "NaN"

    @staticmethod
    def _len(array, **kwargs):
        return len(array)

    @staticmethod
    def _print_header(headers):
        text = []
        underline = []
        for feature in headers:
            text.append(f"{feature:>32}")
            underline.append(f"{'_' * len(feature):>32}")
        print(f"{' ':>9}{''.join(text)}\n{' ':>9}{''.join(underline)}")

    @classmethod
    def _print_desc(cls, data, headers):
        desc_vars = {
            "count": cls._len,
            "mean": cls._mean,
            "std": cls._std_dev,
            "min": cls._min,
            "25%": cls._percentile,
            "50%": cls._percentile,
            "75%": cls._percentile,
            "max": cls._max,
            "unique": cls._unique,
        }
        for k, v in desc_vars.items():
            values = [f"{k:>8}:"]
            for feature in headers:
                try:
                    values.append(f"{v(data[feature], value=k):>32.6f}")  # fuck float
                except ValueError:
                    values.append(f"{v(data[feature], value=k):>32}")
            print("".join(values))

    @classmethod
    def describe(cls, data: pd.DataFrame, headers):
        while headers:
            tmp_header = [headers.pop(0) for _ in range(4) if len(headers)]
            cls._print_header(tmp_header)
            cls._print_desc(data, tmp_header)
            if headers:
                print("\n\n")
