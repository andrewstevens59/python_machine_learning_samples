


 
import time
import datetime
import calendar
import json
import copy

import pickle
import math
import sys

import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import json


'''
  1. Find out from which age (in years) the number of sick patients is higher than the number of healthy patients 
  2. Find out which two features have the highest correlation with height 
  3. Add a new column "BMI" = weight (in kilograms) divided by squared height (in meters). Find out which group of patients has a higher median BMI - the sick ones or the healthy ones 
  4. Find out how many of the patients are female 
  5. Find out which two features have the highest Spearman correlation and plot their values on a 2D plane 
'''

df = pd.read_csv('/Users/andrewstevens/Downloads/cardio_ds.csv', delimeter=';')

print (df.columns)
print (df)
sys.exit(0)

#Index([u'id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio'], dtype='object')

'''
for index, row in df.iterrows():


	df['age']
'''

