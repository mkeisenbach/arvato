# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:05:08 2019

@author: Mei
"""
import sys
import pandas as pd

from sklearn.externals import joblib

# Needed for loaded model
from model import make_model

if __name__ == '__main__':
    ''' Predict probabilities for test data
    
    Args:
        testdata_filepath (str): filepath to cleaned test data
        model_filepath (str): filepath to model
        submission_filename (str): name to be used for output file
    '''
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

    # Create submission
    submission = pd.DataFrame({'LNR':LNR, 'RESPONSE':test_pred[:,1]})
    
    # Write to file
    print('Writing predictions to file...')
    submission.to_csv(submission_filename, index=False)
    
    print('Done')