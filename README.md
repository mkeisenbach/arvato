# arvato

## Installation
To run the Jupyter notebooks and python scripts, you will need a standard installtion of Anaconda with Python 3.6.x

Additional libraries needed:
- sklearn
- imblearn
- keras
- tensorflow (tensorflow-gpu is preferred as the neural network training can take quite a lot of time)

## Project Motivation
This project was done as the capstone requirement for Udacity's Data Scientist Nanodegree. The goal was to characterize what types of individuals are more likely to be customers of a mail-order retailer and predict which customers would respond positively to marketing campaigns.

## Data
The data used for this project not publically available. It was provided only to those participating in the "in class" competition.

## Files
- features.csv - data dictionary

- segmentation/Arvato Project Workbook.ipynb - data expoloration and preprocessing
- segmentation/Customer Segmentation Report.ipynb - analysis of customers
- segmentation/clean_data.py - python script for cleaning the segmentation data
- segmenation/fit_clustering.py - python script for fitting clustering model

- supervised/Supervised Learning Using Keras.ipynb - workbook for classification task
- supervised/clean_data - python script for cleaning classification data
- supervised/fit_classifier.py - python script for fitting the classifier
- supervised/predict.py - python script for making predictions

## Instructions
### Customer Segmentation Report
1. Clean population and customer data

  - From the segmentation directory, run: 
  <pre>python clean_data.py [data_dir]/Udacity_AZDIAS_052018.csv ../features.csv</pre>

  - From the segmentation directory, run: 
  <pre>python clean_data.py [data_dir]/Udacity_CUSTOMERS_052018.csv.csv ../features.csv</pre>

2. Run the Customer Segmentation Jupyter notebook

### Supervised Learning Model
1. Clean the training and test data

- From the supervised directory, run: 
  <pre>python clean_data.py [data_dir]/Udacity_MAILOUT_052018_TRAIN.csv ../features.csv</pre>

- From the supervised directory, 
  <pre>run: python clean_data.py [data_dir]/Udacity_MAILOUT_052018_TEST.csv ../features.csv</pre>

2. Run the script to train the model:

- From the supervised directory, run: 
  <pre>python fit_classifier.py [data_dir]/Udacity_MAILOUT_052018_TRAIN_clean.csv  ../features.csv [model]</pre>

3. Run the script to predict probabilities:

- From the supervised directory, run: 
  <pre>python predict.py [data_dir]/Udacity_MAILOUT_052018_TEST_clean.csv [model].pkl preds.csv</pre>
