import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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
    sns.histplot(data=dataframe, x='temp_max')
    plt.ylabel("Count")
    plt.xlabel("Temperature")
    plt.title("Temperature Max")
    plt.show()


def temp_max_facegrid_lineplot(dataframe: pd.DataFrame):

    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['month'] = dataframe['date'].dt.month
    dataframe['year'] = dataframe['date'].dt.year

    g = sns.FacetGrid(dataframe, col="year")
    g.map_dataframe(sns.lineplot, x="month", y="temp_max")
    g.set_axis_labels("Month", "Max Temperature(°C)")
    plt.show()


def precipitation_facegrid_scatterplot(dataframe: pd.DataFrame):
    """ TODO:
    """
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['month'] = dataframe['date'].dt.month
    dataframe['year'] = dataframe['date'].dt.year

    p = sns.FacetGrid(dataframe, col="year")
    p.map_dataframe(sns.scatterplot, x="month", y="precipitation")
    p.set_axis_labels("Month", "Precipitation")

    w = sns.FacetGrid(dataframe, col="year")
    w.map_dataframe(sns.scatterplot, x="month", y="wind")
    w.set_axis_labels("Month", "Wind Speed")
    plt.show()

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
    # dataset_info(df)
    temp_max_histplot(df)
    # temp_max_facegrid_lineplot(df)
    # precipitation_facegrid_scatterplot(df)


if __name__ == '__main__':
    main()
