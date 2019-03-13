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
- Arvato Report.pdf - Analysis report

### Customer Segmentation
- segmentation/Arvato Project Workbook.ipynb - Data expoloration and preprocessing
- segmentation/Customer Segmentation Report.ipynb - Analysis of customers
- segmentation/Mailout.ipynb - Analysis of mailout data using clustering model
- segmentation/clean_data.py - Python script for cleaning the segmentation data
- segmenation/fit_clustering.py - File containing clustering pipeline function. This can also be used as a standlone script.

### Marketing Predictions
- supervised/Supervised Learning Using Ensemble Methods.ipynb - Classification using ensemble methods 
- supervised/Supervised Learning Using Keras.ipynb - Classification using a neural network
- supervised/clean_data - Python script for cleaning classification data
- supervised/preprocess.py - Python file for preprocessing functions

## Instructions
### Customer Segmentation Report
1. Clean population and customer data

  - From the segmentation directory, run: 
  <pre>python clean_data.py [data_dir]/Udacity_AZDIAS_052018.csv ../features.csv</pre>

  - From the segmentation directory, run: 
  <pre>python clean_data.py [data_dir]/Udacity_CUSTOMERS_052018.csv.csv ../features.csv</pre>

2. Run the Customer Segmentation Report notebook

### Marketing predictions
1. Clean the training and test data
- From the supervised directory, run: 
  <pre>python clean_data.py [data_dir]/Udacity_MAILOUT_052018_TRAIN.csv ../features.csv</pre>

- From the supervised directory, 
  <pre>run: python clean_data.py [data_dir]/Udacity_MAILOUT_052018_TEST.csv ../features.csv</pre>

2. Run the Supervised Learning Using Ensemble Methods notebook

## Results
The detailed analysis of the results can be read in [this Medium post](https://medium.com/@mei.eisenbach/arvato-bertelsmann-customer-analysis-ae1aac59a1ef) or in Arvato Report.pdf.

### Customer segmentation
- One group was found to be more likely to be customers: These indivduals were more religious, older and savers.
- Two groups were found to be less likely to be customers: 1) Individuals with low purchasing activity and wealth (also younger) and 2) Individuals from areas with low population density and were less cultural minded/religiousness

### Marketing predictions
The final model had an auc_roc score of 0.76294 and a Kaggle score of 0.80143 (https://www.kaggle.com/c/udacity-arvato-identify-customers/leaderboard).

Model | Local score |Kaggle Score
--- | --- | ---
Keras | 0.65836 | 0.65842
Gradient Boost | 0.76524 | 0.79327
AdaBoost | 0.76238 | 0.79791
LightGBM (final) | 0.76294 | 0.80143
