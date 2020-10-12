import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=sys.maxsize)


class LogisticRegression:
    @staticmethod
    def _sigmoid_function(x):
        print(f"sigmoid {x}: {1 / (1 + np.exp(-x))}")
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _cost_function(h, y):
        m = len(y)
        print(
            f"cost {h} - {y}: {(1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))}"
        )
        return (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))

    def _gradient_descent(self, X, h, theta, y, m):
        gradient_value = np.dot(X.T, (h - y)) / m
        theta -= self.alpha * gradient_value
        print(f"gradient descent theta = {theta}: {gradient_value} {theta}")
        return theta

    def _plot_cost(self, costh):
        matplotlib.use("pdf")
        figures = []
        for cost, c in costh:
            fig = plt.figure()
            plt.plot(range(len(cost)), cost, "r")
            plt.title("Convergence Graph of Cost Function of type-" + str(c) + " vs All")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Cost")
            figures.append(fig)
        with PdfPages(f"ue.pdf") as pdf:
            for f in figures:
                pdf.savefig(f, bbox_inches="tight")

    def __init__(self, alpha=0.01, n_iteration=100):
        self.alpha = alpha  # value in the object
        self.n_iter = n_iteration

    def fit(self, X, y):
        # This function primarily calculates the optimal theta value using which we predict the future data
        print("Fitting the given dataset.")
        self.theta = []
        self.cost = []
        X = np.insert(X, 0, 1, axis=1)
        print(X)
        m = len(y)
        for i in np.unique(y):
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(X.shape[1])
            cost = []
            for _ in range(self.n_iter):
                # z = np.nan_to_num(X.dot(theta))
                z = X.dot(theta)
                print(f"z = {z}\ntheta = {theta}\n X.shape[1] = {X.shape[1]}\n")
                print(f"z = {type(z)}\ntheta = {type(theta)}\n X.shape[1] = {type(X.shape[1])}\n")
                h = self._sigmoid_function(z)
                theta = self._gradient_descent(X, h, theta, y_onevsall, m)
                print(z, h, theta)
                sys.exit()
                cost.append(self._cost_function(h, y_onevsall))
            self.theta.append((theta, i))
            self.cost.append((cost, i))

    def predict(self, X):
        # this function calls the max predict function to classify the individul feature
        X = np.insert(X, 0, 1, axis=1)
        X_predicted = [
            max((self._sigmoid_function(i.dot(theta)), c) for theta, c in self.theta)[1] for i in X
        ]
        return X_predicted

    def score(self, X, y):
        score = sum(self.predict(X) == y) / len(y)
        return score


# We are reading and processing the data provided

data = pd.read_csv("data/datasets/dataset_train.csv")
# Transposing the data
data_T = data
data_T.columns = [
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
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

y_data = data_T["Hogwarts House"].values  # segregating the label vlue from the feature value.
X = pd.DataFrame(
    data=data_T,
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
)

for h in X.columns:
    X[h] = X[h].apply(lambda x: x / max(X[h]))
X = X.to_numpy()

from sklearn.model_selection import train_test_split

scores = []
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.90)
    print(f"{X_train.shape}")
    logi = LogisticRegression(n_iteration=30000)
    logi.fit(X_train, y_train)
    predition1 = logi.predict(X_test)
    score1 = logi.score(X_test, y_test)
    print("the accuracy of the model is ", score1, logi.cost)
    scores.append(score1)

print(np.mean(scores))
logi._plot_cost(logi.cost)
# Here we ae plotting the Cost value and showing how it is depreciating close to 0 with each iteration
