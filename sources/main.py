from Parser.ArgParser import ArgParser
from Parser.CSVParser import CSVParser


def main():
    args = ArgParser()
    reader = CSVParser(args)
    reader.csv_parser()


if "__main__" == __name__:
    main()