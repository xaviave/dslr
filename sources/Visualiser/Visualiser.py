import logging
import sys

import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


logging.getLogger().setLevel(logging.INFO)


class Visualiser:
    raw_data: pd.DataFrame

    @staticmethod
    def _save_as_pdf(figures):
        try:
            with PdfPages(f"data_visualizer{len(figures)}.pdf") as pdf:
                for f in figures:
                    pdf.savefig(f, bbox_inches="tight")
        except:
            logging.error(f"Error while creating data_visualizer{len(figures)}.pdf")
            sys.exit(-1)

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
        plt.axis("off")
        return fig

    @staticmethod
    def _auto_label(ax, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    @staticmethod
    def _process_bar_data(raw_data, head):
        stat = [
            raw_data.loc[lambda df: df["Hogwarts House"] == house, head]
            for house in ["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"]
        ]
        return {
            "right": [
                (stat[index] == "Right").sum()
                for index in range(4)
            ],
            "left": [
                (stat[index] == "Left").sum()
                for index in range(4)
            ],
        }

    def _histogram_visualizer(self, head):
        hands = self._process_bar_data(self.raw_data, head)
        houses = set(self.raw_data.loc[:, "Hogwarts House"])
        x = np.arange(len(houses))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, hands["right"], width, label="Right")
        rects2 = ax.bar(x + width / 2, hands["left"], width, label="Left")
        ax.set_xticks(x)
        ax.set_xticklabels(houses)
        ax.legend()
        self._auto_label(ax, rects1)
        self._auto_label(ax, rects2)
        fig.tight_layout()
        return fig

    def _scatter_visualizer(self, head):
        fig = plt.figure()
        plt.scatter(
            x=tuple(self.raw_data.loc[:, "Hogwarts House"]),
            y=tuple(self.raw_data.loc[:, head]),
        )
        return fig

    @staticmethod
    def _date_visualizer(x, y):
        fig = plt.figure()
        plt.subplot(x, y)
        return fig

    @staticmethod
    def _update_tab(head, func):
        fig = func(head)
        plt.title(f"Hogwarts House compared to '{head}")
        plt.close()
        return fig

    def _advanced_visualizer(self, header):
        logging.info("Creating tabs in pdf...")
        func = {
            "Best Hand": self._histogram_visualizer,
            "Birthday": self._date_visualizer,
        }
        figures = [self._text_page()]
        for head in header:
            figures.append(
                self._update_tab(head, func.get(head, self._scatter_visualizer))
            )
        self._save_as_pdf(figures)

    def __init__(self, raw_data):
        self.raw_data = raw_data

    def visualizer(self, header):
        if self.raw_data.empty:
            logging.error("raw_data not init")
            sys.exit(-1)
        logging.info(self.raw_data.head())
        matplotlib.use("pdf")
        self._advanced_visualizer(header)
