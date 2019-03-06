# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:19:46 2019

@author: Mei
"""
import sys
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from keras.wrappers.scikit_learn import KerasClassifier

from model import make_model
from preprocessor import make_preprocessor

if __name__ == '__main__':
    ''' Fits the classifier on the train training data
    
    Args:
        cleandata_filepath (str): Filepath to the cleaned training data
        features_filepath (str): Filepath to the feature information
        model_name (str): Name of model to be saved as a pickle file
    '''
    cleandata_filepath, features_filepath, model_name = sys.argv[1:]
    
    # Load feature info
    feat_info = pd.read_csv(features_filepath)
    feat_info.set_index('attribute', inplace=True)
    
    # Load data
    print('Loading data...')
    clean_df = pd.read_csv(cleandata_filepath, sep=';')
    y = clean_df.RESPONSE
    X = clean_df.drop(['RESPONSE', 'LNR'], axis=1)
    
    # Define column types for the preprocessor
    numerical_columns = feat_info[feat_info.type == 'numeric'].index.drop(['GEBURTSJAHR','KBA13_ANZAHL_PKW'])
    categorical_columns = X.columns.drop(numerical_columns)
    
    # Build model
    print('Building model...')
    model = Pipeline([
                ('preprocess', make_preprocessor(numerical_columns, categorical_columns)),
                ('clf', KerasClassifier(build_fn=make_model, verbose=True))
            ])

    
    class_weight = {0:1, 1:15}
    param_grid = {'clf__n_features':[X.shape[1]],
                  'clf__learn_rate':[0.0001],
                  'clf__class_weight':[class_weight],
                  'clf__batch_size':[64],
                  'clf__epochs':[30]}
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)

    # Fit model
    print('Fitting model...')
    grid_result = grid.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Save model
    print('Saving model...')    
    joblib.dump(grid, model_name+'.pkl')
    
    print('Done')
    