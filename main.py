import pandas as pd


def dataset_info(dataframe: pd.DataFrame):
    dataframe.info(verbose=True)
    print("\n")

    print("Maximum values :")
    print(dataframe.max(numeric_only=True))
    print("\n")
    print("Minimum values :")
    print(dataframe.min(numeric_only=True))

    print("\n")

    print("Mean of each numeric value :")
    print(dataframe.mean(numeric_only=True))

    print("\n")

    print("Value counts for each weather type :")
    print(dataframe.weather.value_counts())
    print("Most common weather type :")
    print(dataframe.weather.mode())


def temp_max_histplot(dataframe: pd.DataFrame):
    """ TODO:
    """


def temp_max_facegrid_lineplot(dataframe: pd.DataFrame):
    """ TODO:
    """


def precipitation_facegrid_scatterplot(dataframe: pd.DataFrame):
    """ TODO:
    """


def weather_countplot(dataframe: pd.DataFrame):
    """ TODO:
    """


def weather_piechart(dataframe: pd.DataFrame):
    """ TODO:
    """


def lr_predictor_random_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def lr_predictor_default_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def svr_predictor_default_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def main():
    df = pd.read_csv('seattle-weather.csv')
    dataset_info(df)


if __name__ == '__main__':
    main()
