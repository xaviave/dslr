from Parser.ArgParser import ArgParser
from describe import Describe


def main():
    args = ArgParser()
    describer = Describe(args)
    describer.describe()


if "__main__" == __name__:
    main()
