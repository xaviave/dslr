import pandas as pd
import os
import sys
import logging

logging.getLogger().setLevel(logging.INFO)


def school_scorer():
    try:
        school_data = pd.read_csv(f"{os.path.abspath('dataset_truth.csv')}")[
            "Hogwarts House"
        ].values
        user_data = pd.read_csv(f"{os.path.abspath('houses.csv')}")["Hogwarts House"].values
        compared_data = sum(
            [1 for i in range(1, len(school_data)) if school_data[i] == user_data[i]]
        )
        return compared_data / len(school_data)
    except Exception as e:
        logging.error(e)
        sys.exit(-1)


if "__main__" == __name__:
    print(f"Accuracy of the test: {school_scorer()}")
