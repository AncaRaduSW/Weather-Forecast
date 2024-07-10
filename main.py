import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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
    sns.countplot(dataframe, x="weather")
    plt.xlabel("Weather")
    plt.ylabel("Count")
    plt.title("Weather Type Distribution")
    plt.show()


def weather_piechart(dataframe: pd.DataFrame):
    colors = sns.color_palette('pastel')[0:5]

    data = dataframe.weather.value_counts()
    data.plot.pie(y="weather", autopct='%1.1f%%',colors=colors)

    plt.xlabel("Weather")
    plt.ylabel("Type")

    plt.title("Weather Piechart")
    plt.show()

def lr_predictor_given_split(dataframe: pd.DataFrame, split):
    dates = pd.to_datetime(dataframe['date'])
    dataframe['month'] = dates.dt.month
    dataframe['year'] = dates.dt.year

    X = dataframe.drop(columns=['weather', 'temp_max'], axis=1).dropna()
    y = dataframe['temp_max']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, shuffle=False)

    X_train_values = X_train.drop(columns=['date'])
    X_test_values = X_test.drop(columns=['date'])

    # Splitting the data into training and testing data
    regr = LinearRegression()

    regr.fit(X_train_values, y_train)

    y_pred = regr.predict(X_test_values)

    print("Mean Squared Error : ", mean_squared_error(y_test, y_pred))
    print("R2 score : ", r2_score(y_test, y_pred))

    date_tests = pd.to_datetime(X_test['date'])
    date = np.array(date_tests)

    f = plt.figure()
    f.set_figwidth(13)
    f.set_figheight(6)

    plt.grid(True)

    plt.scatter(date, y_test, color='b', label='Actual', )
    plt.plot(date, y_test, color='b')
    plt.scatter(date, y_pred, color='r', marker='o', label='Predicted')
    plt.plot(date, y_pred, color='r')

    plt.show()

def lr_predictor_random_split(dataframe: pd.DataFrame):

    lr_predictor_given_split(dataframe, random.random())


def lr_predictor_default_split(dataframe: pd.DataFrame):
    lr_predictor_given_split(dataframe, 0.2)


def svr_predictor_default_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def main():
    df = pd.read_csv('seattle-weather.csv')
    # dataset_info(df)
    # temp_max_histplot(df)
    # temp_max_facegrid_lineplot(df)
    # precipitation_facegrid_scatterplot(df)
    # weather_countplot(df)
    # weather_piechart(df)
    lr_predictor_random_split(df)
    lr_predictor_default_split(df)


if __name__ == '__main__':
    main()
