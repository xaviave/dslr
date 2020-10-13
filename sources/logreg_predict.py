from LogReg.LogReg import LogReg
from Parser.ArgParser import ArgParser
from Parser.CSVParser import CSVParser


def run():
    args = ArgParser()
    dataset = CSVParser(args, parse=True)
    log_train = LogReg()
    log_train.load_theta()
    prediction = log_train.predict(dataset.np_df)
    dataset.write_to_csv(prediction)


if "__main__" == __name__:
    run()
