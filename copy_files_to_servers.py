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
import paramiko



print "transferring"
'''
["104.237.4.91", "01+3k-sqJSK9i8J"],
["104.237.4.35", "UnfactRachelMonieKidang"],
["104.237.4.171", "067W@vI0qi-PiY3"],
["104.251.214.63", "0Y1jz9qC*D99M*f"],
["104.237.4.141", "0fP95kA2=Lcg*Q7"],
'''
	
servers = [

	["104.237.4.174", "GemotsUncoopPinotSnack"],
]

for server in servers:
	t = paramiko.Transport((server[0], 22))
	t.connect(username="root", password=server[1])
	sftp = paramiko.SFTPClient.from_transport(t)

	print ("connected", server[0])
	sftp.put("execute_news_signals_summary.py", "/root/trading/execute_news_signals_summary.py")
	print ("here")
	sftp.put("execute_all_update_news_signals.py", "/root/trading/execute_all_update_news_signals.py")
	
	sftp.put("execute_news_signals_plots.py", "/root/trading/execute_news_signals_plots.py")
	
	if server[0] == "104.237.4.174":
		sftp.put("analyze_strategy_return_stat.py", "/root/trading/analyze_strategy_return_stat.py")
		sftp.put("production/train_price_action_model.py", "/root/trading/production/train_price_action_model.py")
		sftp.put("production/execute_basket_stat_arb.py", "/root/trading/production/execute_basket_stat_arb.py")
		sftp.put("production/stat_arb_price_distribution.py", "/root/trading/production/stat_arb_price_distribution.py")
		sftp.put("production/execute_basket_stat_arb_martingale.py", "/root/trading/production/execute_basket_stat_arb_martingale.py")
		sftp.put("production/execute_basket_obv.py", "/root/trading/production/execute_basket_obv.py")
		sftp.put("production/execute_basket_obv_martingale.py", "/root/trading/production/execute_basket_obv_martingale.py")

	if server[0] == "104.237.4.91":
		sftp.put("production/get_price_delta.py", "/root/trading/production/get_price_delta.py")
		sftp.put("execute_update_portfolio.py", "/root/trading/execute_update_portfolio.py")
		sftp.put("execute_update_trade_journal.py", "/root/trading/execute_update_trade_journal.py")
		sftp.put("production/send_trade_alerts.py", "/root/trading/production/send_trade_alerts.py")

	if server[0] == "104.237.4.171":
		sftp.put("production/draw_macro_trend_lines.py", "/root/trading/production/draw_macro_trend_lines.py")
		sftp.put("production/train_basket_model.py", "/root/trading/production/train_basket_model.py")
		sftp.put("production/execute_straddle_trade_setup.py", "/var/www/html/execute_straddle_trade_setup.py")
		sftp.put("production/short_term_economic_forecast.py", "/root/trading/production/short_term_economic_forecast.py")
		
	if server[0] == "104.237.4.35":
		sftp.put("production/train_economic_news_model.py", "/root/trading/production/train_economic_news_model.py")
		sftp.put("execute_news_martingale.py", "/root/trading/execute_news_martingale.py")

	if server[0] == "104.251.214.63":
		sftp.put("production/create_json_news_summary.py", "/root/trading/create_json_news_summary.py")
