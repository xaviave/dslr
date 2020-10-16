import os
import random

import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split

from Tools.LogReg import LogReg
from school_scorer import school_scorer


def get_dataset():
    file_name = os.path.join("data", "datasets", "dataset_test.csv")
    raw_data = pd.read_csv(file_name)
    return pd.DataFrame(
        data=raw_data,
        columns=[
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Charms",
            "Flying",
        ],
    ).to_numpy()


def test():
    scores = []
    timer = []
    for _ in range(10):
        """
        training
        """
        size = random.randint(50, 90) / 100
        start = datetime.now()
        logi = LogReg(train=True, n_iteration=30000)
        x_train, x_test, y_train, y_test = train_test_split(
            logi.df, logi.np_df_train, test_size=size
        )
        logi.np_df = x_train
        logi.np_df_train = y_train
        logi.train(logi.np_df, logi.np_df_train)
        logi.save_theta()
        timer.append(datetime.now() - start)
        """
            Predict
        """
        logi.load_theta()
        test_prediction = logi.predict(x_test)
        logi.write_to_csv("test_houses.csv", test_prediction, ["Hogwarts House"])
        """
            Test
        """
        score = logi.score(x_test, y_test)
        print(
            f"""
            Train sample size {x_train.shape[0]}
            The accuracy of the model is {score}
            """
        )
        scores.append(score)
    print(f"Mean accuracy is {np.mean(scores)}\nMean time is {np.mean(timer)}")


if "__main__" == __name__:
    test()
