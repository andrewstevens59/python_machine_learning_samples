import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

import mysql.connector
import pandas as pd
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from bisect import bisect
import seaborn as sns
import psutil
import logging
import datetime as dt
from uuid import getnode as get_mac
import traceback

cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

barrier_models = ["'M1'", "'M9'", "'M15'", "'M21'"]
regression_models = ["'M3'", "'M6'", "'M12'", "'M18'"]

for pair in currency_pairs:
    for model_types in [barrier_models, regression_models]:

        if model_types == barrier_models:
            print ("barrier_models")
        else:
            print ("regression_models")

    
        for model in model_types:
        
            query = ("""
                SELECT auc, rmse
                FROM model_historical_stats 
                where pair = '{}' 
                and model_type = ({})
                """.format(pair, model))

            cursor = cnx.cursor()
            cursor.execute(query)

            aucs = []
            rmses = []
            for row1 in cursor:
                auc = row1[0]
                rmse = row1[1]

                if auc != 0:
                    aucs.append(auc)

                if rmse != 0:
                    rmses.append(rmse)

            if len(aucs) > 0:
                print (model, pair, "AUC", np.percentile(aucs, 70))

            if len(rmses) > 0:
                print (model, pair, "RMSE", np.percentile(rmses, 30))
     

