import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from random import *
import time
import os.path

from sklearn import datasets, linear_model


from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import numpy as np
from sklearn.decomposition import PCA

import pickle

currency_pairs = {
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
}

currencies = ['USD', 'EUR', 'NZD', 'AUD', 'CAD', 'GBP', 'JPY', 'CHF']

portfolio_map = {}
for currency in currencies:
	portfolio_map[currency] = 0

file = open("currency_portfolio.txt", "r") 
portfolio_text = file.read() 
file.close()

import re
toks = re.split('	', portfolio_text)

total_exposure = 0
for i in range(len(toks)):
	trans_tok = toks[i].replace("/", "_")
	if trans_tok in currency_pairs:

		amount = int(toks[i+1].replace(",", ""))
		if toks[i+2] == 'SHORT':
			amount = -amount

		portfolio_map[trans_tok[0:3]] = portfolio_map[trans_tok[0:3]] + amount
		portfolio_map[trans_tok[4:7]] = portfolio_map[trans_tok[4:7]] - amount
		total_exposure = total_exposure + (abs(amount) * 2)

revert_model = pickle.load(open("/tmp/" + "revert_model", 'rb'))
future_model = pickle.load(open("/tmp/" + "trend_model", 'rb'))

print revert_model

print portfolio_map

class CurrencyResid:

    def __init__(self):
        self.pair = ""
        self.resid = 0
        self.dir = ""


def display_info(pair, revert_model, is_revert):

	currency1 = pair[0:3]
	currency2 = pair[4:7]

	signal1 = revert_model[currency1]
	signal2 = revert_model[currency2]

	diff = signal1 - signal2
	if diff > 0:
		print pair, diff, "SELL"
	else:
		print pair, diff, "BUY"

	'''
	average_revert_ratio = 0
	average_follow_ratio = 0

	average_revert_hold = 0
	average_follow_hold = 0

	signal_follow_num = 0
	signal_revert_num = 0
	signal_num = 0


	for i in range(15):
		sigma_min = 0.5 + (i * 0.1)

		if abs(signal1 - signal2) < sigma_min:
			continue


		key = pair + str(sigma_min)
		if key not in global_revert_pair_ratio_map_past:
			continue

		if key not in global_follow_pair_ratio_map_past:
			continue

		average_revert_ratio = average_revert_ratio + global_revert_pair_ratio_map_past[pair + str(sigma_min)]
		average_follow_ratio = average_follow_ratio + global_follow_pair_ratio_map_past[pair + str(sigma_min)]
		signal_num = signal_num + 1

	revert_side = "BUY"
	follow_side = "SELL"

	if (signal1 - signal2) > 0:
		revert_side = "SELL"
		follow_side = "BUY"

	if signal_num > 0:
		print pair, "Average Revert Signal", float(average_revert_ratio) / signal_num, revert_side
		print pair, "Average Follow Signal", float(average_follow_ratio) / signal_num, follow_side
	'''

for pair in currency_pairs:

	currency1 = pair[0:3]
	currency2 = pair[4:7]

	revert_signal1 = revert_model[currency1]
	revert_signal2 = revert_model[currency2]

	future_signal1 = future_model[currency1]
	future_signal2 = future_model[currency2]

	if (abs(revert_signal1 - revert_signal2) + abs(future_signal1 - future_signal2)) / 2 < 1:
		continue


	print "------------ Past Model ------------"
	display_info(pair, revert_model, False)

	

	print "------------ Future Model ------------"
	display_info(pair, future_model, True)

	print ""
	
