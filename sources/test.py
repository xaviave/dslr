import random

import numpy as np
from sklearn.model_selection import train_test_split

from LogReg.LogReg import LogReg
from Parser.ArgParser import ArgParser
from Parser.CSVParser import CSVParser

scores = []


def test():
    args = ArgParser()
    dataset = CSVParser(args, parse=True, train=True)
    for _ in range(10):
        size = random.randint(20, 90) / 100
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.df, dataset.np_df_train, test_size=size
        )
        logi = LogReg(n_iteration=30000)
        logi.train(x_train, y_train)
        score = logi.score(x_test, y_test)
        print(
            f"Train and test sample size {x_train.shape[0]}\nThe accuracy of the model is {score}"
        )
        scores.append(score)
    print(f"Mean accuracy is {np.mean(scores)}")


if "__main__" == __name__:
    test()
