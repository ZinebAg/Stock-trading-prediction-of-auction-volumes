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



# loading the data:
data_input_training = pd.read_csv("/Users/Mac/Desktop/Spring 2022/Semester Project/data/input_training.csv.gz", compression='gzip')
data_output_training= pd.read_csv("/Users/Mac/Desktop/Spring 2022/Semester Project/data/output_training_IxKGwDV.csv")
y=pd.read_csv("/Users/Mac/Desktop/Spring 2022/Semester Project/data/y_response.csv.gz", compression='gzip')

# helpers

def na_roughfix(x):
  'returns the imputed matrix using median'
  return(x.fillna(x.median()))


# from https://stackoverflow.com/questions/18703136/proximity-matrix-in-sklearn-ensemble-randomforestclassifier

def proximityMatrix(model, X, normalize=True):      

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:,0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:,i]
        proxMat += 1*np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat  

# define the matrix : 

def define_matrix(x_original):
  x_original_bis=na_roughfix(x_original)
  x_original_bis["class"]=0
  b = SLHS.LatinHypercubeSampling(x_original_bis, 10,"selection")
  samples = b.sample_points()
  samples['class']= 1
  frames = [x_original_bis, samples]
  df = pd.concat(frames)
  return(df)


def rfImput_unsupvd(x, iter =10):
  print("--------------------------------------")
  print("we have still {} iteration left".format(iter))
  x_roughfix= na_roughfix(x)
  rf_imput = x.copy()
  while iter:
    # print("--------------------------------------")
    # print("we have still {} iteration left".format(iter))
    df=define_matrix(x_roughfix)
    X,y =df.iloc[:,0:127],df.iloc[:,127]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    x_prox=proximityMatrix(clf, X.iloc[range(x.shape[0]),:], normalize=True)

    # print("x shape is", x.shape )
    # print("x_prox shape is", x.shape )
    # print("x_roughfix shape is", x_roughfix.shape)

    for i in range(rf_imput.shape[1]):
       # print("we are at column nbr",i)
        
        rf_imput.iloc[:,i]=na_fix(x.iloc[:,i],x_roughfix.iloc[:,i],x_prox)
    
    diff_rel=dist_rel(rf_imput,x_roughfix)
    if diff_rel < 1e-5:
      break
    else:
      x_roughfix= rf_imput
      iter= iter-1
  
  return (rf_imput)
    
    
# calculate the relative distance
def dist_rel(x_impute, x_org):
  max_x= x_impute.abs().apply(max,axis=0)
  # if False:
  diff_x = (x_impute- x_org)/max_x
  diff_rel = (diff_x**2).values.sum()/((x_org/max_x)**2).values.sum()
  return(diff_rel)


def na_fix(na_values, rough_values, x_prox):
  if na_values.shape[0] != rough_values.shape[0]:
    print("na_fix and rough_values must have the same length")
    print("na_values",na_values.shape)
    print("rough_values", rough_values.shape)
    sys.exit()
  elif rough_values.shape[0] != x_prox.shape[1]:
    print("x_prox and rough_values have incorrect sizes")
    print("--------------------------------------")
    print("x_prox",x_prox.shape)
    print("--------------------------------------")
    print("rough_values", rough_values.shape)
    sys.exit()
  
  # Na imputation only for na data:
  na_list=na_values.index[na_values.apply(np.isnan)]
  replaced_values= rough_values
  for i in range(len(na_list)):
    j = na_list[i]
    # print("--------------------------------------")
    # print("so j = ", j, "and we have that x_prox has the shape",pd.DataFrame(x_prox).shape )
    # print("--------------------------------------")

    pd.DataFrame(x_prox).iloc[j,j] = 0 # imputed datum itself
    replaced_values[j]= k_Weighted_mean(rough_values, pd.DataFrame(x_prox).iloc[:,j])

  return(replaced_values)


# k: the number of neighbors.
def k_Weighted_mean(value, weight, k=5):
  k = min(k, value.shape[0])
  order_weight =  -np.sort(-weight) 
  ww = weight.iloc[order_weight]
  vv = value.iloc[order_weight]
  ret = (ww[1:k] * vv[1:k]).values.sum()/ ww[1:k].values.sum()
  return(ret)


# to test it for a fixed k and data subset: 

data_trial= data_input_training.sample(10000)
response=  pd.merge(data_trial,data_output_training[data_output_training['ID'].isin(data_trial.ID)], on = 'ID' ).target

#preparing the data
data_trial=data_trial.reset_index()
data_trial=data_trial.iloc[:,1:128]

#
new_data_trial=rfImput_unsupvd(data_trial, iter =10)

# verify if the constraint is well verified
new_data_trial.iloc[:,64:125].sum(axis=1).hist()
