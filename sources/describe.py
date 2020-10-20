from Tools.DatasetHandler import DatasetHandler


def main():
    dataset = DatasetHandler(parse=True)
    dataset.describe()


if "__main__" == __name__:
    main()
