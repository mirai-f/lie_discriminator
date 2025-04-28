import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import nltk
import pymorphy2
from nltk.corpus import stopwords


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatize_flag=True):
        nltk.download('stopwords', quiet=True)
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
        words = [self._lemmatize(word) for word in words if word not in self.russian_stopwords]
        return " ".join(words) if words else "empty"
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame({   
                col: X[col].apply(self._preprocess_text) for col in X.columns
            })
        else:
            return pd.Series(X).apply(self._preprocess_text)

        
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
        
        futures = pd.DataFrame({
            'len': column.str.len(),
            'num_punctuation': column.str.count('[:,.!?]'),
            'num_upper': column.str.count(r'[A-ZА-Я]'),
            'dummy_feature': [0] * len(X)
        })

        return futures[self.features_names]
    
    
def get_features_names_out(preprocessor):

    tfidf_features = (
        preprocessor
        .named_transformers_['text']
        .named_steps['tfidfvectorizer']
        .get_feature_names_out()
    )
    
    feature_extractor_names = (
        preprocessor
        .named_transformers_['features']
        .named_steps['featureextractor']
        .features_names
    )
    
    all_feature_names = np.concatenate([tfidf_features, feature_extractor_names])
    return all_feature_names


def get_features_names_out(preprocessor):
    tfidf_features = (
        preprocessor
        .named_transformers_['text']
        .named_steps['tfidfvectorizer']
        .get_feature_names_out()
    )
    
    # Получаем имена признаков из FeatureExtractor
    feature_extractor_names = (
        preprocessor
        .named_transformers_['features']
        .named_steps['featureextractor']
        .features_names
    )
    
    # Объединяем все имена признаков
    all_feature_names = np.concatenate([tfidf_features, feature_extractor_names])
    return all_feature_names



def build_preprocessor():
    features_pipeline = make_pipeline(
        FeatureExtractor(),
        StandardScaler()
    )

    text_pipeline = make_pipeline(
        TextPreprocessor(),
        TfidfVectorizer()
    )

    preprocessor = ColumnTransformer([
        ('text', text_pipeline, 'title'),
        ('features', features_pipeline, 'title')
    ])

    return preprocessor