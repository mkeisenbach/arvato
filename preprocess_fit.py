# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:01:01 2019

@author: Mei
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_fit(clean_df, pca_n, impute_strat='median'):
    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strat)
    imputed_df = imputer.fit_transform(clean_df)
    
    # Apply feature scaling
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(imputed_df)
    
    # PCA
    pca = PCA(pca_n)
    pca_df = pca.fit_transform(scaled_df)
    
    return (pca_df, imputer, scaler, pca)
    
