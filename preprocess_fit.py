# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:01:01 2019

@author: Mei
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_fit(clean_df, pca_n=0, impute_strat='median'):
    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strat)
    return_df = imputer.fit_transform(clean_df)
    
    # Apply feature scaling
    scaler = StandardScaler()
    return_df = scaler.fit_transform(return_df)
    
    # PCA
    pca = None
    if pca_n > 0:
        pca = PCA(pca_n)
        return_df = pca.fit_transform(return_df)
    
    return (return_df, imputer, scaler, pca)
    
