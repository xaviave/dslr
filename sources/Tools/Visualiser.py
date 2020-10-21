import inspect
import logging
import sys
import matplotlib

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from Tools.ArgParser import ArgParser

logging.getLogger().setLevel(logging.INFO)


class Visualiser(ArgParser):
    raw_data: pd.DataFrame
    houses: np.ndarray

    """
        Override methods
    """

    def _add_exclusive_args(self, parser):
        visualiser_group = parser.add_mutually_exclusive_group(required=False)
        visualiser_group.add_argument(
            "-v",
            "--visualiser",
            action="store_const",
            const={"name": "advanced", "func": self._advanced_visualizer},
            help="Render a tab to visualize data",
            dest="type_visualizer",
        )
        visualiser_group.add_argument(
            "-hh",
            "--histogram",
            action="store_const",
            const={"name": "histogram", "func": self._histogram_visualizer},
            help="Render an histogram for a specific data",
            dest="type_visualizer",
        )
        visualiser_group.add_argument(
            "-sc",
            "--scatter_plot",
            action="store_const",
            const={"name": "scatter", "func": self._scatter_plot_visualizer},
            help="Render a scatter plot graph for a specific data",
            dest="type_visualizer",
        )
        visualiser_group.add_argument(
            "-pp",
            "--pair_plot",
            action="store_const",
            const={"name": "pair", "func": self._pair_plot_visualizer},
            help="Render a pair plot graph for a specific data",
            dest="type_visualizer",
        )

    """
        Private methods
    """

    def _save_as_pdf(self, file_name, figures):
        try:
            with PdfPages(f"{file_name}_{len(figures) - 1}.pdf") as pdf:
                for f in figures:
                    pdf.savefig(f, bbox_inches="tight")
        except Exception as e:
            self._exit(exception=e, message=f"Error while creating {file_name}.pdf")

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

    def _pick_by_elements(self, raw_data, filters, column_filter, feature):
        try:
            return [
                raw_data.loc[raw_data[column_filter] == element, feature] for element in filters
            ]
        except (IndexError, ValueError) as e:
            self._exit(exception=e, message="Error while filtering data for visualiser")

    def _create_histogram(
        self, raw_data: list, title="Histogram", kde_label=None, xlabel="x", ylabel="y", yticks=None
    ):
        fig = plt.figure()
        for index, data in enumerate(raw_data):
            plt.hist(data, density=True, bins=30, alpha=0.5)
            if kde_label is not None:
                mn, mx = plt.xlim()
                plt.xlim(mn, mx)
                kde_xs = np.linspace(mn, mx, 301)
                kde = st.gaussian_kde(data)
                try:
                    plt.plot(kde_xs, kde.pdf(kde_xs), label=kde_label[index])
                except IndexError:
                    self._exit(message="Error while creating histogram")
        plt.title(title)
        plt.legend(kde_label, loc="upper left")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yticks([] if yticks is None else yticks)
        plt.close()
        return fig

    def _histogram_visualizer(self, header):
        figures = [self._text_page()]
        for feature in header:
            filtered_data = self._pick_by_elements(
                self.raw_data, self.houses, "Hogwarts House", feature
            )
            figures.append(
                self._create_histogram(
                    filtered_data,
                    title=f"Homogeneity between the four houses '{feature}",
                    kde_label=self.houses,
                    xlabel="Marks",
                    ylabel="Students",
                )
            )
        self._save_as_pdf("histogram_visualizer", figures)

    def _pair_plot_visualizer(self, head):
        logging.warning(
            f"{inspect.currentframe().f_code.co_name}:Need to be refactor - No hard coded data please"
        )
        self._exit(message="Not implemented")

    def _create_scatter(self, x, y=None, title="Scatter", label=None, xlabel="x", ylabel=None, yticks=None):
        fig = plt.figure()
        for index, data in enumerate(x):
            try:
                plt.scatter(data, data if y is None else y[index], alpha=0.5)
            except IndexError:
                self._exit(message="Error while creating scatter plot")
        plt.title(title)
        plt.legend(label, loc="upper left")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yticks([] if yticks is None else yticks)
        plt.close()
        return fig

    def _scatter_plot_visualizer(self, header):
        figures = [self._text_page()]
        filtered_data = [self.raw_data.loc[:, feature] for feature in header]
        figures.append(
            self._create_scatter(
                filtered_data,
                title="Two similar features",
                label=header,
                xlabel="Marks"
            )
        )
        self._save_as_pdf("scatter_visualizer", figures)

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
        self._save_as_pdf("advanced_visualizer", figures)

    @staticmethod
    def _exit(exception=None, message="Error", mod=-1):
        if exception:
            logging.error(f"{exception}")
        logging.error(f"{message}")
        sys.exit(mod)

    def __init__(self, type_visualizer="advanced", func=_scatter_plot_visualizer):
        super().__init__()
        mod_visualizer = self.get_args(
            "type_visualizer", default_value={"name": type_visualizer, "func": func}
        )
        self.header_visualizer = mod_visualizer.get("name")
        self.func_visualizer = mod_visualizer.get("func")

    """
        Public methods
    """

    def visualizer(self, header):
        matplotlib.use("pdf")
        if self.raw_data.empty:
            self._exit(message="Please init raw_data")
        self.houses = np.unique(self.raw_data.loc[:, "Hogwarts House"])
        self.func_visualizer(header)
