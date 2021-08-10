import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from pytz import timezone
import xgboost as xgb
from datetime import timedelta
import mysql.connector
from lxml.html import fromstring
from itertools import cycle
from scipy import stats
import datetime as dt
import traceback


import re

import time
import datetime
import calendar
from dateutil import tz
import requests
import lxml.html as lh
import json
import copy

import math
import sys
import re

import numpy as np
import pandas as pd 
import pycurl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko
from sklearn.model_selection import train_test_split
import json


import os
import bisect

import paramiko
import json

import logging
import os
import enum

import matplotlib

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import logging
import datetime as dt
from uuid import getnode as get_mac
import traceback



df = pd.read_csv('/Users/andrewstevens/Downloads/diabetes.csv')

print (df)
df['id'] = df['id'].apply(lambda x: int(x))

categorical = ['smoke', 'active', 'gender', 'cholesterol', 'gluc']

#df["pressure_linear"] = df["pressure_linear"].appl() # turn to linear

numeric = list(set(df.columns) - set(categorical)) + ['id']
numeric.remove('pressure')


dummies_df = pd.get_dummies(df[categorical])
dummies_df["id"] = df["id"].values.tolist()

print (dummies_df)

print (df[numeric])


print (df.set_index('id'))

print (dummies_df.set_index('id'))

print (len(df), len(dummies_df))
print (numeric)
print (dummies_df.columns)

dummy_columns = list(set(dummies_df.columns) -set(numeric)) + ['id']

#df = (df[numeric]).set_index('id').join(dummies_df[dummy_columns].set_index('id'), how='inner')



df = df[numeric]


features = list(df.columns)#.remove('diabetes').remove('id')#.remove('pressure')
features.remove('diabetes')
#features.remove('pressure')


clf = xgb.XGBClassifier()

X = df[features].values.tolist()
y = df['diabetes'].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y)


print (len(X_train), len(y_train))

clf.fit(np.array(X_train), y_train)


preds = clf.predict(np.array(X_test))

corrects = sum([1 if y_actual == y_test else 0 for y_actual, y_test in zip(y_test, preds)])

accuracy = float(corrects) / len(X_test)
print (accuracy)

print (corrects)

