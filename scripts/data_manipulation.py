from typing import Callable

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, LabelEncoder
from datetime import datetime
# When importing from notebook
import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join('..')))
    from scripts.logger_creator import CreateLogger
except:
    from logger_creator import CreateLogger

logger = CreateLogger('Data Manipulatior', handlers=1)
logger = logger.get_default_logger()


class DataManipulator:
    def __init__(self, df: pd.DataFrame, deep=False):
        """
            Returns a DataManipulator Object with the passed DataFrame Data set as its own DataFrame
            Parameters
            ----------
            df:
                Type: pd.DataFrame

            Returns
            -------
            None
        """
        if(deep):
            self.df = df.copy(deep=True)
        else:
            self.df = df

    def add_column(self, new_col_name: str, col_1: str, col_2: str, func: Callable) -> None:
        try:
            self.df[new_col_name] = func(self.df[col_1], self.df[col_2])
        except Exception as e:
            print(e)

    def sort_using_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrame sorted with the specified column, default dataframe sorting
            Parameters
            ----------
            column:
                Type: str

            Returns
            -------
            pd.DataFrame
        """
        try:
            return self.df.sort_values(column)
        except:
            print("Failed to sort using the specified column")

    def get_top_sorted_by_column(self, column: str, length: int) -> pd.DataFrame:
        """
            Returns the objects DataFrame sorted in descending order and selecting the top ones with the specified column
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            pre_df = self.df.sort_values(
                column, ascending=False).iloc[:length, :]
            return pd.DataFrame(pre_df.loc[:, column])
        except:
            print("Failed to sort using the specified column and get the top results")

    def scale_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrames column scaled using MinMaxScaler
            Parameters
            ----------
            column:
                Type: str

            Returns
            -------
            pd.DataFrame
        """
        try:
            scale_column_df = pd.DataFrame(self.df[column])
            scale_column_values = scale_column_df.values
            print(
                f'The max and min values of the scaled {column} column are:\n\tmax: {scale_column_df.iloc[:, 0].min()}\n\tmin: {scale_column_df.iloc[:, 0].max()}')
            min_max_scaler = MinMaxScaler()
            scaled_values = min_max_scaler.fit_transform(scale_column_values)
            self.df[column] = scaled_values

            return self.df

        except:
            print("Failed to scale the column")

    def normalize_df(self, sep_index: int) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
            ----------
            None

            Returns
            -------
            pd.DataFrame
        """
        try:
            unnormal = self.df.iloc[:, :sep_index]
            normalized = self.df.iloc[:, sep_index:]
            columns = normalized.columns
            normalizer = Normalizer()
            normalized_data = normalizer.fit_transform(normalized)
            normalized_data = pd.DataFrame(normalized_data, columns=columns)
            self.df = pd.merge(left=unnormal, left_index=True,
                               right=normalized_data, right_index=True, how='inner')

            return self.df

        except:
            print("Failed to normalize the dataframe")

    def standardize_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using StandardScalar
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            std_column_df = pd.DataFrame(self.df[column])
            std_column_values = std_column_df.values
            standardizer = StandardScaler()
            normalized_data = standardizer.fit_transform(std_column_values)
            self.df[column] = normalized_data

            return self.df
        except:
            print("Failed to standardize the column")

    def standardize_columns(self, columns: list) -> pd.DataFrame:
        try:
            for col in columns:
                self.df = self.standardize_column(col)

            return self.df
        except:
            print(f"Failed to standardize {col} column")

    def minmax_scale_column(self, column: str, range_tup: tuple = (0, 1)) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            std_column_df = pd.DataFrame(self.df[column])
            std_column_values = std_column_df.values
            minmax_scaler = MinMaxScaler(feature_range=range_tup)
            normalized_data = minmax_scaler.fit_transform(std_column_values)
            self.df[column] = normalized_data

            return self.df
        except:
            print("Failed to standardize the column")

    def minmax_scale_columns(self, columns: list, range_tup: tuple = (0, 1)) -> pd.DataFrame:
        try:
            for col in columns:
                self.df = self.minmax_scale_column(col, range_tup)

            return self.df
        except:
            print(f"Failed to MinMax standardize {col} column")

    def fill_columns_with_max(self, columns: list) -> None:
        try:
            for col in columns:
                self.df[col] = self.df[col].fillna(self.df[col].max())

        except Exception as e:
            print("Failed to fill with max value")

    def fill_columns_with_most_frequent(self, columns: list) -> None:
        try:
            for col in columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])

        except Exception as e:
            print("Failed to fill with max value")

    def label_columns(self, columns: list) -> dict:
        labelers = {}
        try:
            for col in columns:
                le = LabelEncoder()
                le_fitted = le.fit(self.df[col].values)
                self.df[col] = le_fitted.transform(self.df[col].values)
                labelers[col] = le_fitted

            return labelers

        except Exception as e:
            print("Failed to Label Encode columns")


if __name__ == "__main__":
    df = pd.DataFrame([['abebeb', 'debebe'], ['abebbe', 'asdbjasd']],
                      columns=['abebe', 'debebe'])

    print(df)

    def change(a: pd.Series, b: pd.Series):
        for index, value in a.items():
            return value + b[index]

    dm = DataManipulator(df)
    dm.add_column('new_col', 'abebe', 'debebe', change)

    print(dm.df)
