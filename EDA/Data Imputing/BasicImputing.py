# packages needed: 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline 
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
from sklearn.metrics import mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import idaes.surrogate.pysmo.sampling as SLHS
import sys
random.seed(10)



# first the basic imputings: 
def nan_to_mean(dataset, r_v=True, a_r=True, constrained=False):
    """""
      Dataset : data
      r_v: if true will impute the relative volume. If False, no change is issued
      a_r: if true will impute the absolute revenue. If False, no change is issued
      constrained: if true will take into consideration that the r_v columns should add up to one.
    """

    dataset_copy= dataset.copy()
    if a_r:
      dataset_copy.iloc[:,3:64]=dataset_copy.iloc[:,3:64].fillna(dataset_copy.iloc[:,3:64].mean())
    if r_v:
      if constrained:
        nan_count=dataset_copy.iloc[:,64:125].isna().sum(axis=1) 
        values_to_fill_with=(1-df_train.iloc[:,64:125].sum(axis=1))/nan_count
       # dataset_copy.iloc[:,64:125]=dataset_copy.iloc[:,64:125].fillna(values_to_fill_with)
        for i in range(64,125):
            dataset_copy.iloc[:,i]=np.where(np.isnan(dataset_copy.iloc[:,i]), values_to_fill_with, dataset_copy.iloc[:,i]) 
      else:
        dataset_copy.iloc[:,64:125]=dataset_copy.iloc[:,64:125].fillna(dataset_copy.iloc[:,64:125].mean())

    return(dataset_copy)


def nan_to_median(dataset, r_v=True, a_r=True, constrained=False):
    """""
      Dataset : data
      r_v: if true will impute the relative volume. If False, no change is issued
      a_r: if true will impute the absolute revenue. If False, no change is issued
      constrained: if true will take into consideration that the r_v columns should add up to one.
    """
    dataset_copy= dataset.copy()
    if a_r:
      dataset_copy.iloc[:,3:64]=dataset_copy.iloc[:,3:64].fillna(dataset_copy.iloc[:,3:64].median())

    if r_v:
      if constrained:
        nan_count=dataset_copy.iloc[:,65:125].isna().sum(axis=1) 
        median_values=dataset_copy.iloc[:,65:125].median()
        values_to_fill_with=(1-median_values).apply(lambda a: a/nan_count.replace({0:1}) ).transpose()
        dataset_copy.iloc[:,64:125]=dataset_copy.iloc[:,64:125].fillna(values_to_fill_with)
      else:
        dataset_copy.iloc[:,64:125]=dataset_copy.iloc[:,64:125].fillna(dataset_copy.iloc[:,64:125].median())

    return(dataset_copy)


def add_count_NA(dataset, r_v=True, a_r=True):
    """ 
      counts the number of missing values
      Dataset : data
      r_v: if true will impute the relative volume. If False, no change is issued
      a_r: if true will impute the absolute revenue. If False, no change is issued
    """
    dataset_binary = dataset.copy()

    if a_r:
      dataset_binary["NA_count_abs_ret"]=df_train.iloc[:,3:64].isna().sum(axis=1) 
    if r_v:
      dataset_binary["NA_count_rel_vol"]=df_train.iloc[:,64:125].isna().sum(axis=1) 

    return(dataset_binary)


def nan_bfill(dataset, r_v=True, a_r=True):
    """""
      Dataset : data
      r_v: if true will impute the relative volume. If False, no change is issued
      a_r: if true will impute the absolute revenue. If False, no change is issued
  
    """
    dataset_copy= dataset.copy()
    if a_r:
      dataset_copy.iloc[:,3:64]=dataset_copy.iloc[:,3:64].fillna(method='bfill')

    if r_v:
      dataset_copy.iloc[:,64:125]=dataset_copy.iloc[:,64:125].fillna(method='bfill')

    return(dataset_copy)


def nan_ffill(dataset, r_v=True, a_r=True):
    """""
      Dataset : data
      r_v: if true will impute the relative volume. If False, no change is issued
      a_r: if true will impute the absolute revenue. If False, no change is issued
  
    """
    dataset_copy= dataset.copy()
    if a_r:
      dataset_copy.iloc[:,3:64]=dataset_copy.iloc[:,3:64].fillna(method='ffill')

    if r_v:
      dataset_copy.iloc[:,64:125]=dataset_copy.iloc[:,64:125].fillna(method='ffill')

    return(dataset_copy)


