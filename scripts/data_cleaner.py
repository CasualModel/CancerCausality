import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, df: pd.DataFrame, deep=False) -> None:
        """
        Returns a DataCleaner Object with the passed DataFrame Data set as its own DataFrame
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

    def remove_unwanted_columns(self, columns: list) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns in the list are removed
        Parameters
        ----------
        columns:
            Type: list

        Returns
        -------
        pd.DataFrame
        """
        self.df.drop(columns, axis=1, inplace=True)
        return self.df

    def remove_nulls(self) -> pd.DataFrame:
        return self.df.dropna()

    def change_columns_type_to(self, cols: list, data_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns data types are changed to the specified data type
        Parameters
        ----------
        cols:
            Type: list
        data_type:
            Type: str

        Returns
        -------
        pd.DataFrame
        """
        try:
            for col in cols:
                self.df[col] = self.df[col].astype(data_type)
        except:
            print('Failed to change columns type')

        return self.df

    def remove_single_value_columns(self, unique_value_counts: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame where columns with a single value are removed
        Parameters
        ----------
        unique_value_counts:
            Type: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        drop_cols = list(
            unique_value_counts.loc[unique_value_counts['Unique Value Count'] == 1].index)
        return self.df.drop(drop_cols, axis=1, inplace=True)

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Returns a DataFrame where duplicate rows are removed
        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
        """
        removables = self.df[self.df.duplicated()].index
        return self.df.drop(index=removables, inplace=True)

    def fill_numeric_values(self, missing_cols: list, acceptable_skewness: float = 5.0) -> pd.DataFrame:
        """
        Returns a DataFrame where numeric columns are filled with either median or mean based on their skewness
        Parameters
        ----------
        missing_cols:
            Type: list
        acceptable_skewness:
            Type: float
            Default value = 5.0

        Returns
        -------
        pd.DataFrame
        """
        df_skew_data = self.df[missing_cols]
        df_skew = df_skew_data.skew(axis=0, skipna=True)
        for i in df_skew.index:
            if(df_skew[i] < acceptable_skewness and df_skew[i] > (acceptable_skewness * -1)):
                value = self.df[i].mean()
                self.df[i].fillna(value, inplace=True)
            else:
                value = self.df[i].median()
                self.df[i].fillna(value, inplace=True)

        return self.df

    def add_columns_from_another_df_using_column(self, from_df: pd.DataFrame, base_col: str, add_columns: list) -> pd.DataFrame:
        try:
            new_df = self.df.copy(deep=True)
            from_df.sort_values(base_col, ascending=True, inplace=True)
            for col in add_columns:
                col_index = from_df.columns.tolist().index(col)
                new_df[col] = new_df[base_col].apply(
                    lambda x: from_df.iloc[x-1, col_index])

            return new_df

        except:
            print('Failed to add columns from other dataframe')

    def fill_non_numeric_values(self, missing_cols: list, ffill: bool = True, bfill: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame where non-numeric columns are filled with forward or backward fill
        Parameters
        ----------
        missing_cols:
            Type: list
        ffill:
            Type: bool
            Default value = True
        bfill:
            Type: bool
            Default value = False

        Returns
        -------
        pd.DataFrame
        """
        for col in missing_cols:
            if(ffill == True and bfill == True):
                self.df[col].fillna(method='ffill', inplace=True)
                self.df[col].fillna(method='bfill', inplace=True)

            elif(ffill == True and bfill == False):
                self.df[col].fillna(method='ffill', inplace=True)

            elif(ffill == False and bfill == True):
                self.df[col].fillna(method='bfill', inplace=True)

            else:
                self.df[col].fillna(method='bfill', inplace=True)
                self.df[col].fillna(method='ffill', inplace=True)

        return self.df

    def fix_outlier(self, column: str) -> pd.DataFrame:
        """
        Returns a DataFrame where outlier of the specified column is fixed
        Parameters
        ----------
        column:
            Type: str

        Returns
        -------
        pd.DataFrame
        """
        self.df[column] = np.where(self.df[column] > self.df[column].quantile(
            0.95), self.df[column].median(), self.df[column])

        return self.df

    def fix_outlier_columns(self, columns: list) -> pd.DataFrame:
        """
        Returns a DataFrame where outlier of the specified columns is fixed
        Parameters
        ----------
        columns:
            Type: list

        Returns
        -------
        pd.DataFrame
        """
        try:
            for column in columns:
                self.df[column] = np.where(self.df[column] > self.df[column].quantile(
                    0.95), self.df[column].median(), self.df[column])
        except:
            print("Cant fix outliers for each column")

        return self.df

    def standardized_column(self, columns: list, new_name: list, func) -> pd.DataFrame:
        """
        Returns a DataFrame where specified columns are standardized based on a given function and given new names after
        Parameters
        ----------
        columns:
            Type: list
        new_name:
            Type: list
        func:
            Type: function

        Returns
        -------
        pd.DataFrame
        """
        try:
            assert(len(columns) == len(new_name))
            for index, col in enumerate(columns):
                self.df[col] = func(self.df[col])
                self.df.rename(columns={col: new_name[index]}, inplace=True)

        except AssertionError:
            print('size of columns and names provided is not equal')

        except:
            print('standardization failed')

        return self.df

    def optimize_df(self) -> pd.DataFrame:
        """
        Returns the DataFrames information after all column data types are optimized (to a lower data type)
        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
        """
        data_types = self.df.dtypes
        optimizable = ['float64', 'int64']
        try:
            for col in data_types.index:
                if(data_types[col] in optimizable):
                    if(data_types[col] == 'float64'):
                        # downcasting a float column
                        self.df[col] = pd.to_numeric(
                            self.df[col], downcast='float')
                    elif(data_types[col] == 'int64'):
                        # downcasting an integer column
                        self.df[col] = pd.to_numeric(
                            self.df[col], downcast='unsigned')

            return self.df

        except:
            print('Failed to optimize')

    def save_clean_data(self, name: str):
        """
        The objects dataframe gets saved with the specified name 
        Parameters
        ----------
        name:
            Type: str

        Returns
        -------
        None
        """
        try:
            self.df.to_csv(name, index=False)

        except:
            print("Failed to save data")
