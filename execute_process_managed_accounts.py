import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.ensemble import GradientBoostingRegressor
import download_calendar as download_calendar
from sklearn.model_selection import cross_val_score
import gzip, cPickle
import string
import random as rand

import os
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from maximize_sharpe import *

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from evaluate_model import evaluate
from uuid import getnode as get_mac
import socket
import paramiko
import json

import delta_process as delta_process
import breakout as breakout_process
import volatility_process as volatility_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import logging
from close_trade import *
import os

import mysql.connector


def sendCurlRequest(url, request_type, post_data = None):
    response_buffer = StringIO()
    header_buffer = StringIO()
    curl = pycurl.Curl()

    curl.setopt(curl.URL, url)

    curl.setopt(pycurl.CUSTOMREQUEST, request_type)

    curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

    curl.setopt(curl.HEADERFUNCTION, header_buffer.write)


    if post_data != None:
        print post_data
        curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer ' + api_key, 'Content-Type: application/json'])
        curl.setopt(pycurl.POSTFIELDS, post_data)
    else:
        curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer ' + api_key])

    curl.perform()
    curl.close()

    response_value = response_buffer.getvalue()
    header_value = header_buffer.getvalue()

    return response_value, header_value


print "here"
cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

cursor = cnx.cursor()
query = ("SELECT * FROM managed_accounts where status=0")

cursor.execute(query)

setup_rows = []
for row1 in cursor:
    setup_rows.append(row1)

cursor.close()

for row in setup_rows:

    print row

    account_nbr = row[1]
    api_key = row[2]

    found = False
    for account_type in ["fxpractice", "fxtrade"]:
        response_value, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_nbr + "/summary", "GET")
        j = json.loads(response_value)

        if u'errorMessage' in j:
            continue

        found = True
        account_profit = float(j['account'][u'unrealizedPL'])
        account_balance = float(j['account'][u'balance'])
        margin_available = float(j['account']['marginAvailable'])
        margin_used = float(j['account']['marginUsed'])

        print str(account_type == "fxpractice")
        cursor = cnx.cursor()
        query = ("UPDATE managed_accounts SET \
            is_hedged = '" + ('1' if j['account'][u'hedgingEnabled'] else '0') + "', \
            account_value = '" + str(account_balance + account_profit) + "', \
            is_demo = '" + ('1' if account_type == "fxpractice" else '0') + "', \
            currency = '" + j['account'][u'currency'] + "', \
            status = '2' \
            WHERE account_nbr = '" + account_nbr + "' and \
            api_key = '" + api_key + "' \
            ")
        cursor.execute(query)
        cnx.commit()
        cursor.close()

    if found == False:
        cursor = cnx.cursor()
        query = ("UPDATE managed_accounts SET \
            status = '1' \
            WHERE account_nbr = '" + account_nbr + "' and \
            api_key = '" + api_key + "' \
            ")

        cursor.execute(query)
        cnx.commit()
        cursor.close()

cursor = cnx.cursor()
query = ("SELECT * FROM managed_accounts where status=2")

cursor.execute(query)

setup_rows = []
for row1 in cursor:
    setup_rows.append(row1)

cursor.close()

print len(setup_rows)

for row in setup_rows:

    user_id = row[0]
    account_nbr = row[1]
    api_key = row[2]
    is_demo = row[3]

    if is_demo:
        account_type = "fxpractice"
    else:
        account_type = "fxtrade"

    response_value, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_nbr + "/summary", "GET")
    j = json.loads(response_value)

    if u'errorMessage' in j:
        continue

    account_profit = float(j['account'][u'unrealizedPL'])
    account_balance = float(j['account'][u'balance'])

    print "Update", str(account_balance + account_profit) 

    cursor = cnx.cursor()
    query = ("UPDATE managed_accounts SET \
        account_value = '" + str(account_balance + account_profit) + "' \
        WHERE account_nbr = '" + account_nbr + "' and \
        user_id = '" + str(user_id) + "' \
        ")
    cursor.execute(query)
    cnx.commit()
    cursor.close()




