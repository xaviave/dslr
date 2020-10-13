from Tools.ArgParser import ArgParser
from Tools.DatasetHandler import DatasetHandler


def main():
    dataset = DatasetHandler()
    dataset.describe()


if "__main__" == __name__:
    main()
