import re
import logging

import pandas as pd

from math import sqrt, floor, ceil

logging.getLogger().setLevel(logging.INFO)


class Describer:
    """
    Take a dataset in parameter and can display some information about it.
    """

    @staticmethod
    def _percentile(**kwargs):
        # can reduce sorting time by getting already sorted array
        array = kwargs.get("array").sort_values(kind="mergesort").values
        try:
            percent = re.match(r"^(?P<perc>\d{2})%$", kwargs.get("value")).group("perc")
            k = (len(array) - 1) * float(int(percent) / 100)
            f = floor(k)
            c = ceil(k)
            if f == c:
                return array[int(k)]
            return (array[int(f)] * (c - k)) + (array[int(c)] * (k - f))
        except (ValueError, TypeError):
            return "NaN"

    @staticmethod
    def _unique(**kwargs):
        unique = {}
        for value in kwargs.get("array"):
            if value not in unique:
                unique[value] = True
        return len(unique)

    @staticmethod
    def _max(**kwargs):
        maximum = kwargs.get("array")[0]
        for value in kwargs.get("array"):
            if value > maximum:
                maximum = value
        return maximum

    @staticmethod
    def _min(**kwargs):
        minimum = kwargs.get("array")[0]
        for value in kwargs.get("array"):
            if value < minimum:
                minimum = value
        return minimum

    @staticmethod
    def _variance(**kwargs):
        n = len(kwargs.get("array"))
        try:
            mean = sum(kwargs.get("array")) / n
        except (TypeError, ZeroDivisionError):
            return "NaN"
        return sum((x - mean) ** 2 for x in kwargs.get("array")) / n

    @staticmethod
    def _std_dev(**kwargs):
        array_size = len(kwargs.get("array"))
        try:
            mean = sum(kwargs.get("array")) / array_size
            variance = sum((x - mean) ** 2 for x in kwargs.get("array")) / array_size
            return sqrt(variance)
        except (TypeError, ZeroDivisionError):
            return "NaN"

    @staticmethod
    def _mean(**kwargs):
        try:
            return sum(kwargs.get("array")) / len(kwargs.get("array"))
        except (TypeError, ZeroDivisionError):
            return "NaN"

    @staticmethod
    def _len(**kwargs):
        return len(kwargs.get("array"))

    @staticmethod
    def _print_header(headers):
        text = []
        underline = []
        for feature in headers:
            text.append(f"{feature:>33}")
            underline.append(f"{'_' * len(feature):>33}")
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
                    values.append(f"{v(array=data.get(feature), value=k):>33.6f}")
                except ValueError:
                    values.append(f"{v(array=data.get(feature), value=k):>33}")
            print("".join(values))

    @classmethod
    def describe(cls, data: pd.DataFrame, headers):
        while headers:
            tmp_header = [headers.pop(0) for _ in range(4) if len(headers)]
            cls._print_header(tmp_header)
            cls._print_desc(data, tmp_header)
            if headers:
                print("\n\n")
