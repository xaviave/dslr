import os
import sys
import PyQt5
import logging
import argparse
import datetime
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


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
    # better if we take this from a file
    ANALYZED_HEADER: np.ndarray = [
        # "Birthday",
        "Best Hand",  # need to change left and right to binary value
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
        if "Birthday" in self.df.columns:
            self.df["Birthday"] = self.df["Birthday"].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
            )

    def _save_as_pdf(self, figures):
        with PdfPages(f"data_visalizer{len(figures)}.pdf") as pdf:
            for f in figures:
                pdf.savefig(f)

    def _scatter_visualizer(self, head):
        fig = plt.figure()
        plt.scatter(
            x=tuple(self.raw_data.loc[:, "Hogwarts House"]),
            y=tuple(self.df.loc[:, head]),
        )
        return fig

    @staticmethod
    def _date_visualizer(x, y):
        fig = plt.figure()
        plt.subplot(x, y)
        return fig

    @staticmethod
    def _autolabel(ax, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    def _bar_visualizer(self, head):
        houses = set(self.raw_data.loc[:, "Hogwarts House"])
        stat = [
            self.raw_data.loc[lambda df: df["Hogwarts House"] == "Slytherin", head],
            self.raw_data.loc[lambda df: df["Hogwarts House"] == "Ravenclaw", head],
            self.raw_data.loc[lambda df: df["Hogwarts House"] == "Gryffindor", head],
            self.raw_data.loc[lambda df: df["Hogwarts House"] == "Hufflepuff", head],
        ]
        hands = {
            "right": [
                (stat[0] == "Right").sum(),
                (stat[1] == "Right").sum(),
                (stat[2] == "Right").sum(),
                (stat[3] == "Right").sum(),
            ],
            "left": [
                (stat[0] == "Left").sum(),
                (stat[1] == "Left").sum(),
                (stat[2] == "Left").sum(),
                (stat[3] == "Left").sum(),
            ],
        }
        x = np.arange(len(houses))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, hands["right"], width, label="Right")
        rects2 = ax.bar(x + width / 2, hands["left"], width, label="Left")
        ax.set_xticks(x)
        ax.set_xticklabels(houses)
        ax.legend()
        self._autolabel(ax, rects1)
        self._autolabel(ax, rects2)
        fig.tight_layout()
        return fig

    @staticmethod
    def _text_page():
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.text(
            0.5,
            0.5,
            "Data visualisation PDF",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
            color="red",
            transform=ax.transAxes,
        )
        ax.text(
            0.75,
            0.05,
            "by Xamartin and Lotoussa",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            color="black",
            transform=ax.transAxes,
        )
        return fig

    def _update_tab(self, head, func):
        print(head)
        fig = func(head)
        plt.title(f"Hogwarts House compared to '{head}")
        plt.close()
        return fig

    def _advanced_visualizer(self):
        logging.info("Creating tabs in pdf...")
        func = {
            "Best Hand": self._bar_visualizer,
            "Birthday": self._date_visualizer,
        }
        figures = [self._text_page()]
        for head in self.ANALYZED_HEADER:
            figures.append(self._update_tab(head, func.get(head, self._scatter_visualizer)))
        self._save_as_pdf(figures)

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def visualizer(self):
        logging.info(self.df.head())
        if "visu" in self.args.options.keys():
            matplotlib.use("pdf")
            self._advanced_visualizer()
        # need to create the data visializer with graph and numpy
        pass

    def csv_parser(self):
        self._get_csv_file()
        self._as_df()
        self.visualizer()
