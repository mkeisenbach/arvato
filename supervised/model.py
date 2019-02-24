# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 09:39:58 2019

@author: Mei
"""

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

def make_model(n_features, learn_rate):
    '''Creates a Keras sequential model
    
    Args:
        n_features (int): number of input features
        learn_rate (float)
        
    Returns:
        model (keras.models.Sequential)
    '''
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