from itba_ml.utils.logging_helper import get_logger
import scipy.stats as st
import pandas as pd
import numpy as np
from typing import Dict, List


LOGGER = get_logger(
    name=__name__,
    level='INFO'
)


class DataCleaner:

    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        # Define self.df
        self.df: pd.DataFrame = df

    def drop_unuseful_columns(
        self,
        drop_columns: List[str] = None
    ) -> pd.DataFrame:
        # Drop columns with all null values
        self.df.dropna(axis=1, how="all", inplace=True)

        # Drop columns with all values equal to 0
        self.df = self.df.loc[:, (self.df!=0).any(axis=0)]

        # Drop drop_columns
        if drop_columns is not None:
            self.df.drop(columns=drop_columns, errors="ignore", inplace=True)
    
    def remove_duplicates(self) -> pd.DataFrame:
        # Remove duplicated rows
        self.df = self.df.loc[~self.df.index.duplicated(keep='first')]

        # Remove duplicated columns
        self.df = self.df.loc[:, ~self.df.columns.duplicated(keep='first')]
    
    def drop_outliers(self) -> pd.DataFrame:
        # Find numerical columns
        num_cols = list(self.df.select_dtypes(include=['number']).columns)

        # Calculate the absolute z_scores
        z_scores = np.abs(st.zscore(self.df[num_cols]))

        # Remove values where z_score is over 2.5 stdev
        for col in num_cols:
            self.df.loc[z_scores[col] > 2.5] = np.nan
    
    def fill_null_values(
        self,
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        # Find numerical columns
        num_cols = list(self.df.select_dtypes(include=['number']).columns)

        # Interpolate missing values, based on method
        if method == 'interpolate':
            # Interpolate missing values from remaining rows
            self.df[num_cols] = self.df[num_cols].interpolate(method="linear")

        elif method == 'mean':
            # Replace outliers with mean values for that column
            means_dict = {col: self.df[col].mean() for col in num_cols}

            self.df.fillna(value=means_dict, inplace=True)

        elif method == 'median':
            # Replace outliers with median values for that column
            medians_dict = {col: self.df[col].median() for col in num_cols}

            self.df.fillna(value=medians_dict, inplace=True)

        # Fill categorical features with mode
        cat_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
        modes_dict = {col: self.df[col].mode() for col in cat_cols}

        self.df.fillna(value=modes_dict, inplace=True)
    
    def rename_columns(
        self,
        rename_dict: Dict[str, str]
    ) -> pd.DataFrame:
        # Rename columns
        self.df.rename(columns=rename_dict, inplace=True)