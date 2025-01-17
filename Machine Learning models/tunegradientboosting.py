# -*- coding: utf-8 -*-
"""TuneGradientBoosting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ho5Mr8GWIjCl-8t0jthRp5fDoKXJ5XLN
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd 
import matplotlib.pyplot as plt
# %matplotlib inline 
import numpy as np
import matplotlib.animation as animation
import sys
import math
import seaborn as sns
import datetime
import random
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import sys

import re

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor as RFR

random.seed(10)

#train data to run the model

y=pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/output_training_IxKGwDV.csv")
y=y.target

#y=pd.DataFrame(y)

#reading the dataframes with the different imputings
data_RF_KNN = pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/data_RF_KNN_2.csv.gz", compression='gzip')

#data_RF_KNN["target"]=y

# scaling the train dataset
data_RF_KNN.iloc[:,4:127] =scale(data_RF_KNN.iloc[:,4:127])

data_RF_KNN

"""# GRADIENT BOOSTING"""

! pip install xgboost

import xgboost as xgb

target = 'target'
IDcol = 'ID'
predictors = [x for x in data_RF_KNN.columns if x not in [target, IDcol]]

from sklearn.metrics import mean_squared_error

# tuning
# - booster [default=gbtree]
# - eta [default=0.3] between 0.01-0.2
# - max_depth [default=6] between 3-10
# - lambda [default=1 : l2 regularisation default 1 
# - eta L1 regularization term on weight (analogous to Lasso regression)
# eval_metric : rmse

# here will try with samples since working with the whole model is too much to run, with multiple cv, will increase the sample sized and compare

"""# trying with 5000 samples"""

#read
y=pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/output_training_IxKGwDV.csv")
y=y.target
data_RF_KNN = pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/data_RF_KNN_2.csv.gz", compression='gzip')
# join
data_RF_KNN["target"]=y
#scale
data_RF_KNN=data_RF_KNN.sample(5000)
# scaling the train dataset
data_RF_KNN.iloc[:,4:127] =scale(data_RF_KNN.iloc[:,4:127])

# Tune nbr of trees
param_test0 = {
 'n_estimators':range(50,200,20)
}
gsearch0 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=136, max_depth=3,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test0, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch0.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch0.best_params_, gsearch0.best_score_

#  Tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch_1 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=110, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch_1.fit(data_RF_KNN[predictors],data_RF_KNN[target])

gsearch_1.best_params_

gsearch_1.best_score_

# Tune gamma

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=110, max_depth=3,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch3.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch3.best_params_, gsearch3.best_score_

#Tune subsample and colsample_bytree
param_test4 = {
 'colsample_bytree':[i/10.0 for i in range(7,10)],
 'subsample':[i/10.0 for i in range(7,10)]
}
gsearch4 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=110, max_depth=3,
 min_child_weight=5, gamma=0.4, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch4.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch4.best_params_, gsearch3.best_score_

# resulting model, we can now lower the learning rate
tuned_model_5000_samples = xgb.XGBRegressor(
 learning_rate =0.1,
 n_estimators=110,
 max_depth=3,
 min_child_weight=5,
 gamma=0.4,
 subsample=0.9,
 colsample_bytree=0.9,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

"""# trying with 10'000 samples"""

#read
y=pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/output_training_IxKGwDV.csv")
y=y.target
data_RF_KNN = pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/data_RF_KNN_2.csv.gz", compression='gzip')
# join
data_RF_KNN["target"]=y
#scale
data_RF_KNN=data_RF_KNN.sample(10000)
# scaling the train dataset
data_RF_KNN.iloc[:,4:127] =scale(data_RF_KNN.iloc[:,4:127])

# Tune nbr of trees
param_test0 = {
 'n_estimators':range(50,200,20)
}
gsearch0 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=136, max_depth=3,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test0, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch0.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch0.best_params_, gsearch0.best_score_

#  Tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch_1 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=190, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch_1.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch_1.best_params_, gsearch_1.best_score_

# Tune gamma

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=190, max_depth=3,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch3.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch3.best_params_, gsearch3.best_score_

#Tune subsample and colsample_bytree
param_test4 = {
 'colsample_bytree':[i/10.0 for i in range(7,10)],
 'subsample':[i/10.0 for i in range(7,10)]
}
gsearch4 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=110, max_depth=3,
 min_child_weight=3, gamma=0.4, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch4.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch4.best_params_, gsearch3.best_score_

# resulting model, we can now lower the learning rate
tuned_model_10000_samples = xgb.XGBRegressor(
 learning_rate =0.1,
 n_estimators=190,
 max_depth=3,
 min_child_weight=3,
 gamma=0.4,
 subsample=0.9,
 colsample_bytree=0.9,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

"""# trying with 100'000 samples"""

#read
y=pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/output_training_IxKGwDV.csv")
y=y.target
data_RF_KNN = pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/data_RF_KNN_2.csv.gz", compression='gzip')
# join
data_RF_KNN["target"]=y
#scale
data_RF_KNN=data_RF_KNN.sample(100000)
# scaling the train dataset
data_RF_KNN.iloc[:,4:127] =scale(data_RF_KNN.iloc[:,4:127])

# Tune nbr of trees
param_test0 = {
 'n_estimators':range(150,200,10)
}
gsearch0 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=136, max_depth=3,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test0, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch0.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch0.best_params_, gsearch0.best_score_

#  Tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch_1 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=190, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch_1.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch_1.best_params_, gsearch_1.best_score_

# Tune gamma

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=190, max_depth=3,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch3.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch3.best_params_, gsearch3.best_score_

#Tune subsample and colsample_bytree
param_test4 = {
 'colsample_bytree':[i/10.0 for i in range(7,10)],
 'subsample':[i/10.0 for i in range(7,10)]
}
gsearch4 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=190, max_depth=3,
 min_child_weight=5, gamma=0.4, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='neg_mean_absolute_error',n_jobs=4, cv=5)
gsearch4.fit(data_RF_KNN[predictors],data_RF_KNN[target])
gsearch4.best_params_, gsearch3.best_score_

# resulting model, we can now lower the learning rate
tuned_model_100000_samples = xgb.XGBRegressor(
 learning_rate =0.1,
 n_estimators=190,
 max_depth=,
 min_child_weight=,
 gamma=,
 subsample=,
 colsample_bytree=,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

"""# TEST

"""

# resulting model, we can now lower the learning rate
tuned_model_5000_samples = xgb.XGBRegressor(
 learning_rate =0.1,
 n_estimators=110,
 max_depth=3,
 min_child_weight=5,
 gamma=0.4,
 subsample=0.9,
 colsample_bytree=0.9,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

# resulting model, we can now lower the learning rate
tuned_model_10000_samples = xgb.XGBRegressor(
 learning_rate =0.1,
 n_estimators=190,
 max_depth=3,
 min_child_weight=3,
 gamma=0.4,
 subsample=0.9,
 colsample_bytree=0.9,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

# resulting model, we can now lower the learning rate
tuned_model_100000_samples = {}

# reading the test models
test_data_RF_KNN = pd.read_csv("/content/drive/MyDrive/Master Semester Project/data/test_RF_KNN_2.csv.gz", compression='gzip')

submission_file_template= pd.read_csv("/content/drive/MyDrive/Master Semester Project/notebooks/Week6/submission_csv_file_random_example.csv")

#--------------------------------
#_test_data_RF_KNN
#--------------------------------

print("data_RF_KNN")
model_1 = tuned_model_5000_samples
model_1.fit(data_RF_KNN, y)

test_predict_RF_KNN_XGB= tuned_model_5000_samples.predict(test_data_RF_KNN)

RF_KNN_XGB=submission_file_template.copy()
RF_KNN_XGB.target=test_predict_RF_KNN_XGB

RF_KNN_XGB=RF_KNN_XGB.set_index('ID')
RF_KNN_XGB.to_csv("/content/drive/MyDrive/Master Semester Project/notebooks/Week6/submissions/Scale_RF_KNN_XGB_tuned_5000.csv")

# looking at the regression fit:
train_predict = model_1.predict(data_RF_KNN)
print("mse = ", mean_squared_error(train_predict,y))
print("R2 = ", r2_score(y,train_predict))

print("data_RF_KNN trial 2 ")
model_1 = tuned_model_10000_samples
model_1.fit(data_RF_KNN, y)

test_predict_RF_KNN_XGB= tuned_model_10000_samples.predict(test_data_RF_KNN)

RF_KNN_XGB=submission_file_template.copy()
RF_KNN_XGB.target=test_predict_RF_KNN_XGB

RF_KNN_XGB=RF_KNN_XGB.set_index('ID')
RF_KNN_XGB.to_csv("/content/drive/MyDrive/Master Semester Project/notebooks/Week6/submissions/Scale_RF_KNN_XGB_tuned_10000.csv")

# looking at the regression fit:
train_predict = model_1.predict(data_RF_KNN)
print("mse = ", mean_squared_error(train_predict,y))
print("R2 = ", r2_score(y,train_predict))



