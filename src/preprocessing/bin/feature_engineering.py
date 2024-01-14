import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)
import pandas as pd
import numpy as np


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    One hot encoding and ordinal encoding of categorical features

    Parameters
    ----------
    cols_to_one_hot_encode : list of str, default=None
        List of columns to one hot encode
    cols_to_ordinal_encode : list of str, default=None
        List of columns to ordinal encode

    Attributes
    ----------
    one_hot_encoders : list of OneHotEncoder
        List of OneHotEncoder objects
    ordinal_encoders : list of OrdinalEncoder
        List of OrdinalEncoder objects
    """

    def __init__(self, cols_to_one_hot_encode=None, cols_to_ordinal_encode=None):
        self.cols_to_one_hot_encode = cols_to_one_hot_encode
        self.cols_to_ordinal_encode = cols_to_ordinal_encode
        self.one_hot_encoders = (
            [OneHotEncoder(drop="if_binary") for col in cols_to_one_hot_encode]
            if cols_to_one_hot_encode is not None
            else None
        )
        self.ordinal_encoders = (
            [OrdinalEncoder() for col in cols_to_ordinal_encode]
            if cols_to_ordinal_encode is not None
            else None
        )

    def fit(self, X, y=None):
        if self.cols_to_one_hot_encode is not None:
            for i, col in enumerate(self.cols_to_one_hot_encode):
                self.one_hot_encoders[i].fit(X[[col]])
        if self.cols_to_ordinal_encode is not None:
            for i, col in enumerate(self.cols_to_ordinal_encode):
                self.ordinal_encoders[i].fit(X[[col]])
        return self

    def transform(self, X):
        logging.info("Encoding categorical features...")
        if self.cols_to_one_hot_encode is not None:
            for i, col in enumerate(self.cols_to_one_hot_encode):
                encoded = self.one_hot_encoders[i].transform(X[[col]]).toarray()
                X = pd.concat(
                    [
                        X,
                        pd.DataFrame(
                            encoded,
                            columns=self.one_hot_encoders[i].get_feature_names_out(
                                [col]
                            ),
                        ),
                    ],
                    axis=1,
                )
                X = X.drop(columns=[col])
        if self.cols_to_ordinal_encode is not None:
            for i, col in enumerate(self.cols_to_ordinal_encode):
                encoded = self.ordinal_encoders[i].transform(X[[col]])
                X = pd.concat([X, pd.DataFrame(encoded, columns=[col + "2"])], axis=1)
                X = X.drop(columns=[col])

        # Getting rid of spaces in column names
        X.columns = X.columns.str.replace(" ", "_")
        return X


class CyclicScaler(BaseEstimator, TransformerMixin):
    """
    Cyclical encoding of features

    Parameters
    ----------
    cols_to_scale : list of str, default=None
        List of columns to cyclical encode
    """

    def __init__(self, cols_to_scale=None):
        self.cols_to_scale = cols_to_scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("Encoding cyclical features...")
        if self.cols_to_scale is not None:
            for col in self.cols_to_scale:
                X[col + "_sin"] = np.sin(2 * np.pi * X[col] / X[col].max())
                X[col + "_cos"] = np.cos(2 * np.pi * X[col] / X[col].max())
                X = X.drop(col, axis=1)
        return X


class ColumnAdder(BaseEstimator, TransformerMixin):
    """
    Class for adding new features
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("Adding new features...")
        X["total_guests"] = X["no_of_adults"] + X["no_of_children"]
        X["total_price"] = X["avg_price_per_room"] * (
            X["no_of_week_nights"] + X["no_of_weekend_nights"]
        )
        X["free_booking"] = (X["total_price"] == 0).astype(int)
        return X


class Scaler(BaseEstimator, TransformerMixin):
    """
    Class for scaling numerical features

    Parameters
    ----------
    cols_to_scale : list of str, default=None
        List of columns to scale
    scaler : str, default="standard"
        Type of scaler to use. Can be "standard" or "minmax"

    Attributes
    ----------
    scaler : StandardScaler or MinMaxScaler
        Scaler object
    """

    def __init__(self, cols_to_scale=None, scaler="standard"):
        self.cols_to_scale = cols_to_scale
        self.scaler = StandardScaler() if scaler == "standard" else MinMaxScaler()

    def fit(self, X, y=None):
        if self.cols_to_scale is not None:
            self.scaler.fit(X[self.cols_to_scale])
        return self

    def transform(self, X):
        logging.info("Scaling features...")
        if self.cols_to_scale is not None:
            X[self.cols_to_scale] = self.scaler.transform(X[self.cols_to_scale])
        return X


class TargetTransformer(BaseEstimator, TransformerMixin):
    """
    Class for transforming target column into binary and reordering columns

    Parameters
    ----------
    target_col : str, default=None
        Name of the target column
    """

    def __init__(self, target_col=None):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.target_col] = X[self.target_col].map({"Canceled": 1, "Not_Canceled": 0})
        target = X[self.target_col]
        X = pd.concat([X.drop(columns=[self.target_col]), target], axis=1).rename(
            columns={0: self.target_col}
        )
        return X
