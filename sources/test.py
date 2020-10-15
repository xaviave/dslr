import random

import numpy as np
from sklearn.model_selection import train_test_split

from Tools.LogReg import LogReg
from Tools.ArgParser import ArgParser
from Tools.DatasetHandler import DatasetHandler

scores = []


def test():
    for _ in range(10):
        size = random.randint(20, 90) / 100
        logi = LogReg(train=True, n_iteration=30000)
        x_train, x_test, y_train, y_test = train_test_split(
            logi.df, logi.np_df_train, test_size=size
        )
        logi.np_df = x_train
        logi.np_df_train = y_train
        logi.train(logi.np_df, logi.np_df_train)
        logi.save_theta()
        logi.load_theta()
        prediction = logi.predict(x_test)
        logi.write_to_csv(prediction, ["Hogwarts House"])
        score = logi.score(x_test, y_test)
        print(
            f"Train and test sample size {x_train.shape[0]}\n"
            f"The accuracy of the model is {score}"
        )
        scores.append(score)
    print(f"Mean accuracy is {np.mean(scores)}")


if "__main__" == __name__:
    test()
