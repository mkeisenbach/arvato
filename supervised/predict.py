# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:05:08 2019

@author: Mei
"""
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib

if __name__ == '__main__':
    testdata_filepath, model_filepath, submission_filename = sys.argv[1:]
    
    # Load model
    print('Loading model...')
    model = joblib.load(model_filepath)
    
    # Load test data
    print('Loading test data...')
    test = pd.read_csv(testdata_filepath, sep=';')
    LNR = test.LNR
    test.drop('LNR', axis=1, inplace=True)
    
    # Predict probabilities
    print('Making predictions...')
    test_pred = model.predict_proba(test)

    # Convert probabilities to labels    
    threshold = .135
    y_class = np.where(test_pred[:,1] > threshold, 1, 0)
    
    # Create submission
    submission = pd.DataFrame({'LNR':LNR, 'RESPONSE':y_class})
    
    # Write to file
    print('Writing predictions to file...')
    submission.to_csv(submission_filename, index=False)