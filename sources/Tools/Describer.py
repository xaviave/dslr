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
            if type(value) is str and value.isalpha():
                return "NaN"
            if value > maximum:
                maximum = value
        return maximum

    @staticmethod
    def _min(**kwargs):
        minimum = kwargs.get("array")[0]
        for value in kwargs.get("array"):
            if type(value) is str and value.isalpha():
                return "NaN"
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
        text = underline = f"{' ':>9}"
        for feature in headers:
            text += f"{feature:>33}"
            underline += f"{'_' * len(feature):>33}"
        print(f"{text}\n" f"{underline}")

    @staticmethod
    def _print_desc(desc_vars, data, headers):
        for k, v in desc_vars.items():
            text = f"{k:>8}:"
            for feature in headers:
                try:
                    text += f"{v(array=data.get(feature), value=k):>33.6f}"
                except ValueError:
                    text += f"{v(array=data.get(feature), value=k):>33}"
            print(text)

    @classmethod
    def describe(cls, data: pd.DataFrame, headers, slice_print: int):
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
        while headers:
            tmp_header = [headers.pop(0) for _ in range(slice_print) if len(headers)]
            cls._print_header(tmp_header)
            cls._print_desc(desc_vars, data, tmp_header)
            if headers:
                print("\n\n")
