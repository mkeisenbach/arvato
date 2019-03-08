# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:23:40 2019

@author: Mei
"""

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def make_preprocessor(numerical_columns, categorical_columns):
    '''
    Args:
        numerical_columns: list of numerical columns
        categorical_columns: list of categorical columns
    
    Returns:
        preprocessor (sklearn.compose.ColumnTransformer)
    '''
    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler())
    
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'))
    
    preprocessor = ColumnTransformer(
        [('numerical_preprocessing', numerical_pipeline, numerical_columns),
         ('categorical_preprocessing', categorical_pipeline, categorical_columns)],
        remainder='drop')
    
    return preprocessor

def get_columns(df_columns, feat_info):
    '''Get numerical and categorical columns based on the dictionary.
    Columns are checked against dataframe columns so that the returned list
    does not containing any missing columns.
    
    Args:
        df_columns: list of dataframe columns
        feat_info: data dictionary
        
    Returns:
        numerical_columns: list of numerical columns
        categorical_columns: list of categorical columns        
    '''
    numerical_columns = feat_info[feat_info.type == 'numeric']
    numerical_columns = set(numerical_columns).intersection(set(df_columns))
    
    categorical_columns = set(df_columns).difference(numerical_columns)

    return list(numerical_columns), list(categorical_columns)

