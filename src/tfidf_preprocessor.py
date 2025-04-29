import re

import nltk
import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_extractors import FeatureExtractor


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatize_flag=True):
        nltk.download("stopwords", quiet=True)
        self.lemmatize_flag = lemmatize_flag
        self.russian_stopwords = set(stopwords.words("russian"))
        self.morph = pymorphy2.MorphAnalyzer() if lemmatize_flag else None

    def _lemmatize(self, word):
        if not self.lemmatize_flag or not self.morph:
            return word
        parsed = self.morph.parse(word)
        return parsed[0].normal_form if parsed else word

    def _preprocess_text(self, text):
        text = text.lower()
        words = re.findall(r"\w+", text)
        words = [
            self._lemmatize(word)
            for word in words
            if word not in self.russian_stopwords
        ]
        return " ".join(words) if words else "empty"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                {col: X[col].apply(self._preprocess_text) for col in X.columns}
            )
        else:
            return pd.Series(X).apply(self._preprocess_text)


def build_tfidf_preprocessor():
    text_pipeline = make_pipeline(
        TextPreprocessor(), 
        TfidfVectorizer()
    )

    features_pipeline = make_pipeline(
        FeatureExtractor(), 
        StandardScaler()
        )

    tfidf_preprocessor = ColumnTransformer([
        ("text", text_pipeline, "title"), 
        ("features", features_pipeline, "title")
    ])

    return tfidf_preprocessor


def get_features_names_out(preprocessor):
    tfidf_features = (
        preprocessor.named_transformers_["text"]
        .named_steps["tfidfvectorizer"]
        .get_feature_names_out()
    )

    feature_extractor_names = (
        preprocessor.named_transformers_["features"]
        .named_steps["featureextractor"]
        .features_names
    )

    all_feature_names = np.concatenate([tfidf_features, feature_extractor_names])
    return all_feature_names
