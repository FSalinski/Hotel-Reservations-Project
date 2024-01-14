from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(self.cols_to_drop, axis=1)
        return X


class DateCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[(X["arrival_date"] == 29) & (X["arrival_month"] == 2)] = X[
            (X["arrival_date"] == 29) & (X["arrival_month"] == 2)
        ].replace({"arrival_date": {29: 1}, "arrival_month": {2: 3}})
        return X
