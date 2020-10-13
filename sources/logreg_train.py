from LogReg.LogReg import LogReg
from Parser.ArgParser import ArgParser
from Parser.CSVParser import CSVParser


def run():
    args = ArgParser()
    dataset = CSVParser(args, parse=True, train=True)
    log_train = LogReg(n_iteration=30000)
    log_train.train(dataset.np_df, dataset.np_df_train)
    log_train.save_theta()


if "__main__" == __name__:
    run()
