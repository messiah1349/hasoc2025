import os
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._lemmatize_text)

    def _lemmatize_text(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

class TextClassifier:
    def __init__(self, label_column_name: str, output_path: Path, tfidf_max_features: Optional[int] = 1000, 
                 logistic_regression_c: float = 1.0):
        self.label_column_name = label_column_name
        self.output_path = output_path
        self.tfidf_max_features = tfidf_max_features
        self.logistic_regression_c = logistic_regression_c
        self.pipeline = Pipeline([
            ('preprocessor', TextPreprocessor()),
            ('vectorizer', TfidfVectorizer(max_features=self.tfidf_max_features)),
            ('classifier', LogisticRegression(C=self.logistic_regression_c))
        ])
        self.label_encoder = LabelEncoder()

    def fit(self, df: pd.DataFrame):
        X = df['text']
        y = self.label_encoder.fit_transform(df[self.label_column_name])
        self.pipeline.fit(X, y)
        self._save_model_components()

    def transform(self, df: pd.DataFrame):
        self._load_model_components()
        X = df['text']
        return self.pipeline.predict(X)

    def _save_model_components(self):
        os.makedirs(self.output_path, exist_ok=True)
        with open(self.output_path / f'{self.label_column_name}_pipeline.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)
        with open(self.output_path / f'{self.label_column_name}_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

    def _load_model_components(self):
        with open(self.output_path / f'{self.label_column_name}_pipeline.pkl', 'rb') as f:
            self.pipeline = pickle.load(f)
        with open(self.output_path / f'{self.label_column_name}_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

