import numpy as np

from Parser.ArgParser import ArgParser
from Parser.CSVParser import CSVParser


class LogReg(CSVParser):
    cost: list
    theta: list
    #
    # def _plot_cost(self, costh):
    #     matplotlib.use("pdf")
    #     figures = []
    #     for cost, c in costh:
    #         fig = plt.figure()
    #         plt.plot(range(len(cost)), cost, "r")
    #         plt.title(
    #             "Convergence Graph of Cost Function of type-" + str(c) + " vs All"
    #         )
    #         plt.xlabel("Number of Iterations")
    #         plt.ylabel("Cost")
    #         figures.append(fig)
    #     with PdfPages(f"ue.pdf") as pdf:
    #         for f in figures:
    #             pdf.savefig(f, bbox_inches="tight")

    @staticmethod
    def _sigmoid_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _cost_function(h, y):
        m = len(y)
        return (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))

    def _gradient_descent(self, X, h, theta, y, m):
        gradient_value = np.dot(X.T, (h - y)) / m
        theta -= self.alpha * gradient_value
        return theta

    def __init__(self, args: ArgParser):
        super().__init__(args)
        self.csv_parser()
        # self.alpha = vars(args.args).get("alpha")
        # self.n_iter = vars(args.args).get("n_iteration")
        self.alpha = 0.01
        self.n_iter = 1000

    def fit(self):
        # This function primarily calculates the optimal theta value using which we predict the future data
        print("Fitting the given dataset.")
        print(self.df.to_numpy().shape)
        X = np.insert(self.df.to_numpy(), 0, 1, axis=1)
        m = len(self.df_train)
        for i in np.unique(self.df_train):
            y_onevsall = np.where(self.df_train == i, 1, 0)
            theta = np.zeros(X.shape[1])
            cost = []
            for _ in range(self.n_iter):
                z = X.dot(theta)
                h = self._sigmoid_function(z)
                theta = self._gradient_descent(X, h, theta, y_onevsall, m)
                cost.append(self._cost_function(h, y_onevsall))
            self.theta.append((theta, i))
            self.cost.append((cost, i))

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        X_predicted = [
            max((self._sigmoid_function(i.dot(theta)), c) for theta, c in self.theta)[1]
            for i in X
        ]
        return X_predicted

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
