# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 21:19:06 2019

@author: mei
"""

import numpy as np
import pandas as pd
import sys
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def build_model(pca_n, n_clusters):

    pipeline = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scale', StandardScaler()),
            ('pca', PCA(pca_n)),
            ('kmeans', KMeans(n_clusters=n_clusters))
    ])
    
    return pipeline

if __name__ == '__main__':
    
    cleandata_filepath, pca_n, n_clusters = sys.argv[1:]
    
    print('Loading data...')
    clean_df = pd.read_csv(cleandata_filepath, sep=';')
    
    print('Building model...')
    model = build_model(pca_n, n_clusters)
    
    print('Fitting model...')
    model.fit(clean_df)

    print('Saving model...')
    f = open('clust_model' + str(n_clusters) + '.pkl', 'wb')
    pickle.dump(model, f)    

    