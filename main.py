from sources.ArgParser import ArgParser
from sources.CSVParser import CSVParser


def main():
    args = ArgParser()
    reader = CSVParser(args)
    reader.csv_parser()


if "__main__" == __name__:
    main()
