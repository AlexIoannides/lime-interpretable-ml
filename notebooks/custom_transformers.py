"""
custom_transformers.py
~~~~~~~~~~~~~~~~~~~~~~

This module contains custom SkiKit-Learn transfomers for use
in the data preparation pipeline used in this data science 
case study.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class DataFrameAdapter(BaseEstimator, TransformerMixin):
    """DataFrameAdapter
    
    Class for mapping column-subsets of Pandas DataFrames
    to raw Numpy arrays.
    """
    def __init__(self, col_names):
        self.col_names = list(col_names)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.col_names].values
    
    def get_feature_names(self):
        return self.col_names


class CategoricalFeatureEncoder(BaseEstimator, TransformerMixin):
    """CategoricalFeatureEncoder
    
    Class for automating the process of applying 
    one-hot-encoding to all categorical variables in a Numpy
    array of only categorical variables.
    """
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        num_vars = X.shape[1]
        encoded_vars = [self.__transform_single_cat_var__(var) for var in X.T]
        feature_names, features = list(zip(*encoded_vars))
        self.feature_names = list(np.concatenate(feature_names, axis=0))
        return np.concatenate(features, axis=1)

    def get_feature_names(self):
        return self.feature_names
    
    def __transform_single_cat_var__(self, cat_feature_col):
        feature_names_, int_factors = np.unique(cat_feature_col, return_inverse=True)
        one_hot_encoder = OneHotEncoder()
        encoded_factors = one_hot_encoder.fit_transform(int_factors.reshape((-1, 1)))
        return (feature_names_, encoded_factors.toarray())

    
def get_pipline_union_contents(union_pl):
    for pl in union_pl.transformer_list:
        for tf in pl[1].named_steps:
            print('{} : {}'.format(pl[0], tf))
    return None


def get_transformer(union_pl, nested_pl_name, trans_name):
    pipeline = [p[1] for p in union_pl.transformer_list if p[0] == nested_pl_name]
    transformer = pipeline[0].named_steps[trans_name]
    return transformer


def get_feature_names(union_pl, nested_pl_name, trans_name):
    transformer = get_transformer(union_pl, nested_pl_name, trans_name)
    return transformer.get_feature_names()