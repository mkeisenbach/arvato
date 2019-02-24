# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:19:46 2019

@author: Mei
"""
import sys
import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam

# Parse missing_or_known string into a list
def parse_missing(s):
    a = s[1:-1].split(',')
    return a

def build_preprocessor(feat_info, numerical_columns, categorical_columns):
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


def make_model(n_features, learn_rate):
    model = Sequential()
    model.add(Dense(150, input_shape=(n_features,),
              kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(75, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(25, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.10))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=learn_rate),
                  metrics=['acc'])

    return model


if __name__ == '__main__':

    cleandata_filepath, features_filepath, model_name = sys.argv[1:]
    
    # Load feature info
    feat_info = pd.read_csv(features_filepath)
    feat_info.set_index('attribute', inplace=True)
    
    # Load data
    print('Loading data...')
    clean_df = pd.read_csv(cleandata_filepath, sep=';')
#    clean_df.drop('Unnamed: 0', axis=1, inplace=True)
    y = clean_df.RESPONSE
    X = clean_df.drop(['RESPONSE', 'LNR'], axis=1)
    
    # Build preprocessor
    numerical_columns = feat_info[feat_info.type == 'numeric'].index.drop(['GEBURTSJAHR','KBA13_ANZAHL_PKW'])
    categorical_columns = X.columns.drop(numerical_columns)
    
    # Build model
    print('Building model...')
    model = Pipeline([
                ('preprocess', build_preprocessor(feat_info, numerical_columns, categorical_columns)),
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
    
    