# arvato

## Installation
To run the Jupyter notebooks and python scripts, you will need a standard installtion of Anaconda with Python 3.6.x

Additional libraries needed:
- sklearn
- imblearn
- keras
- tensorflow (tensorflow-gpu is preferred as the neural network training can take quite a lot of time)

## Data
The data used for this project not publically available. It was provided only to those participating in an "in class" competition.
	
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
