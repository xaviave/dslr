from Parser.ArgParser import ArgParser
from Describer.Describer import Describer


def main():
    args = ArgParser()
    describer = Describer(args)
    describer.describe()


if "__main__" == __name__:
    main()
