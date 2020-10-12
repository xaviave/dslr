from sklearn.preprocessing import StandardScaler

from LogReg.LogReg import LogReg
from Parser.ArgParser import ArgParser


def run():
    args = ArgParser()
    logreg = LogReg(args)

    scaler = StandardScaler()
    logreg.df_train = scaler.fit_transform(logreg.df_train)
    logreg.fit()


if "__main__" == __name__:
    run()
"""
from sklearn.model_selection import train_test_split

scores = []
for _ in range(10):
    predition1 = logi.predict(X_test)
    score1 = logi.score(X_test, y_test)
    print("the accuracy of the model is ", score1)
    scores.append(score1)

print(np.mean(scores))
# logi._plot_cost(logi.cost)
# Here we ae plotting the Cost value and showing how it is depreciating close to 0 with each iteration
"""
