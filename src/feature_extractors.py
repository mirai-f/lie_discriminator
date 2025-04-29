import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, features_names=["dummy_feature"]):
        self.features_names = features_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            column = X
        elif isinstance(X, pd.DataFrame):
            column = X.iloc[:, 0]
        else:
            column = pd.Series(X)

        futures = pd.DataFrame(
            {
                "len": column.str.len(),
                "num_punctuation": column.str.count("[:,.!?]"),
                "num_upper": column.str.count(r"[A-ZА-Я]"),
                "dummy_feature": [0] * len(X),
                "num_digits": column.str.count(r"\d"),
                "mean_word_len": column.apply(
                    lambda x: np.mean([len(w) for w in x.split()])
                ),
            }
        )

        return futures[self.features_names]
