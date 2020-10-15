import inspect
import logging
import sys

import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from Tools.ArgParser import ArgParser

logging.getLogger().setLevel(logging.INFO)


class Visualiser(ArgParser):
    raw_data: pd.DataFrame

    """
        Override methods
    """

    @staticmethod
    def _exiting(exception=None, message="Error", mod=-1):
        if exception:
            logging.error(f"{exception}\n")
        logging.error(f"{message}")
        sys.exit(mod)

    def _add_exclusive_args(self, parser):
        visualiser_group = parser.add_mutually_exclusive_group(required=False)
        visualiser_group.add_argument(
            "-v",
            "--visualiser",
            action="store_const",
            const=self._advanced_visualizer,
            help="Render a tab to visualize data",
            dest="type_visualizer",
        )
        visualiser_group.add_argument(
            "-hh",
            "--histogram",
            action="store_const",
            const=self._histogram_visualizer,
            help="Render an histogram for a specific data",
            dest="type_visualizer",
        )
        visualiser_group.add_argument(
            "-sc",
            "--scatter_plot",
            action="store_const",
            const=self._scatter_plot_visualizer,
            help="Render a scatter plot graph for a specific data",
            dest="type_visualizer",
        )
        visualiser_group.add_argument(
            "-pp",
            "--pair_plot",
            action="store_const",
            const=self._pair_plot_visualizer,
            help="Render a pair plot graph for a specific data",
            dest="type_visualizer",
        )

    """
        Private methods
    """

    @staticmethod
    def _save_as_pdf(figures):
        try:
            with PdfPages(f"data_visualizer{len(figures)}.pdf") as pdf:
                for f in figures:
                    pdf.savefig(f, bbox_inches="tight")
        except Exception as e:
            logging.error(f"{e}\nError while creating data_visualizer{len(figures)}.pdf")
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
        logging.warning(
            f"{inspect.currentframe().f_code.co_name}:Need to be refactor - No hard coded data please"
        )
        stat = [
            raw_data.loc[lambda df: df["Hogwarts House"] == house, head]
            for house in ["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"]
        ]
        return {
            "right": [(stat[index] == "Right").sum() for index in range(4)],
            "left": [(stat[index] == "Left").sum() for index in range(4)],
        }

    def _histogram_visualizer(self, head):
        logging.warning(
            f"{inspect.currentframe().f_code.co_name}:Need to be refactor - No hard coded data please"
        )
        hands = self._process_bar_data(self.raw_data, head)
        houses = set(self.raw_data.loc[:, "Hogwarts House"])
        x = np.arange(len(houses))
        width = 0.35
        fig, ax = plt.subplots()
        # print(hands)
        rects1 = ax.bar(x - width / 2, hands["right"], width, label="Right")
        rects2 = ax.bar(x + width / 2, hands["left"], width, label="Left")
        ax.set_xticks(x)
        ax.set_xticklabels(houses)
        ax.legend()
        self._auto_label(ax, rects1)
        self._auto_label(ax, rects2)
        fig.tight_layout()
        return fig

    def _pair_plot_visualizer(self, head):
        logging.warning(
            f"{inspect.currentframe().f_code.co_name}:Need to be refactor - No hard coded data please"
        )
        self._exiting(message="Not implemented")

    def _scatter_plot_visualizer(self, head):
        logging.warning(
            f"{inspect.currentframe().f_code.co_name}:Need to be refactor - No hard coded data please"
        )
        fig = plt.figure()
        plt.scatter(
            x=tuple(self.raw_data.loc[:, head]), y=tuple(self.raw_data.loc[:, "Hogwarts House"])
        )
        return fig

    @staticmethod
    def _date_visualizer(x, y):
        logging.warning(
            f"{inspect.currentframe().f_code.co_name}:Need to be refactor - No hard coded data please"
        )
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
        logging.warning(
            f"{inspect.currentframe().f_code.co_name}:Need to be refactor - No hard coded data please"
        )
        logging.info("Creating tabs in pdf...")
        func = {"Best Hand": self._histogram_visualizer, "Birthday": self._date_visualizer}
        figures = [self._text_page()]
        for head in header:
            figures.append(self._update_tab(head, func.get(head, self._scatter_plot_visualizer)))
        self._save_as_pdf(figures)

    def __init__(self, func=_scatter_plot_visualizer):
        # could add different matplotlib backend | for now to much work
        super().__init__()
        self.visualizer_func = self.get_args("type_visualizer", func)

    """
        Public methods
    """

    def visualizer(self, header):
        matplotlib.use("pdf")
        if self.raw_data.empty:
            self._exiting(message="Please init raw_data")
        self.visualizer_func(header)
