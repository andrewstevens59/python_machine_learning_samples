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

import socket
import sys
import time

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
import gzip, cPickle
import string
import random as rand

from os import listdir
from os.path import isfile, join

import os
from sklearn.cluster import SpectralClustering
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
import time
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

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

class Order:

	def __init__(self):
		self.pair = ""
		self.dir = 0
		self.open_price = 0
		self.open_time = 0
		self.amount = 0
		self.PnL = 0
		self.tp_price = 0
		self.sl_price = 0
		self.actual_amount = 0
		self.account_number = None
		self.time_diff_hours = 0
		self.order_id = 0
		self.base_amount = 0
		self.sequence_id = 0
		self.model_key = None
		self.prediction_key = None
		self.margin_used = 0
		self.open_prediction = 0
		self.curr_prediction = 0
		self.portfolio_wt = 0
		self.calendar_time = 0
		self.barrier_size = 0
		self.carry_cost = 0
		self.ideal_price = 0

def sendCurlRequest(api_key, url, request_type, post_data = None):
	response_buffer = StringIO()
	header_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, url)

	curl.setopt(pycurl.CUSTOMREQUEST, request_type)

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.setopt(curl.HEADERFUNCTION, header_buffer.write)

	#2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8 actual

	#3168adee62ceb8b5f750efcec83c2509-1db7870026de5cccb33b220c100a07ab demo

	#16736a148307bf5d91f9f03dd7c91623-af6a25cf6249c9083f11593ad3899f89 demo2

	print url


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

def close_order(order, api_key):

	order_info, _ =  sendCurlRequest(api_key, "https://api-fxpractice.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		return True

	return False

def get_open_trades(api_key, account_number, order_tag):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/" + order_tag + "?count=50"

	while next_link != None:
		response_value, header_value = sendCurlRequest(api_key, next_link, "GET")

		lines = header_value.split('\n')
		lines = filter(lambda line: line.startswith('Link:'), lines)

		if len(lines) > 0:
			line = lines[0]
			next_link = line[line.find("<") + 1 : line.find(">")]
		else:
			next_link = None

		j = json.loads(response_value)
		open_orders = j[order_tag]

		for trade in open_orders:
			order_id = trade['id']
			open_price = float(trade[u'price'])
			
			
			if 'currentUnits' in trade:
				amount = float(trade[u'currentUnits'])
				pair = trade[u'instrument']
				pair = trade['instrument'].replace("/", "_")
				PnL = float(trade['unrealizedPL'])
				margin_used = float(trade['marginUsed'])

				trade_id = order_id
			else:
				amount = 0
				pair = None
				PnL = None
				margin_used = None
				trade_id = trade[u'tradeID']


			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair
			order.dir = amount > 0
			order.order_id = order_id
			order.trade_id = trade_id
			order.account_number = account_number
			order.order_tag = order_tag
			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders

signals1 = json.loads(open("/var/www/html/calendar_output.json", "rb").read())
signals2 = json.loads(open("/var/www/html/z_score_calendar_output.json", "rb").read())

avg_pair_barrier_num = {}
avg_pair_barrier_denom = {}
for model in signals1["new_releases"] + signals2["new_releases"]:  

	if model["pair"] not in avg_pair_barrier_num and len(model["barriers"]) > 0:
		avg_pair_barrier_num[model["pair"]] = 0
		avg_pair_barrier_denom[model["pair"]] = 0


	for barrier in model["barriers"]:

		if barrier["prob"] > 0.5:
			avg_pair_barrier_num[model["pair"]] += barrier["barrier"] * abs(barrier["prob"] - 0.5) * barrier["auc"]
		else:
			avg_pair_barrier_num[model["pair"]] -= barrier["barrier"] * abs(barrier["prob"] - 0.5) * barrier["auc"]

		avg_pair_barrier_denom[model["pair"]] += abs(barrier["prob"] - 0.5) * barrier["auc"]

print avg_pair_barrier_num
print avg_pair_barrier_denom



cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='66.33.203.187',
                              database='newscapital')
cursor = cnx.cursor()

query = ("SELECT * FROM trade_setups")


cursor.execute(query)

setup_rows = []
for row in cursor:
	setup_rows.append(row)

cursor.close()

closed_setup = set()

order_map = {}
stop_loss_map = {}
for row in setup_rows:
	api_key = row[0]
	pair = row[1]
	stop_loss = row[2]
	open_price = row[3]
	max_positions = row[4]
	max_exposure = row[5]
	sizing_factor = row[6]
	spacing_factor = row[7]
	trade_dir = row[8]
	take_profit = row[9]
	account_number = row[10]

	total_pnl = row[11]
	total_orders = row[12]
	total_margin_used = row[13]
	last_update_time = row[14]
	min_profit = row[15]
	barrier_in_pips = row[16]

	if pair not in avg_pair_barrier_num:
		continue

	if api_key + "_" + pair in closed_setup:
		continue

	if api_key + account_number not in order_map:
		orders = get_open_trades(api_key, account_number, "trades")
		order_map[api_key + account_number] = orders

		trade_id_map = {}
		for order in orders:
			 trade_id_map[order.trade_id] = order

		pending = get_open_trades(api_key, account_number, "orders")


		for order in pending:
			trade_id_map[order.trade_id].stop_loss = order.open_price
			stop_loss_map[trade_id_map[order.trade_id].pair + "_" + api_key] = order.open_price
	else:
		orders = order_map[api_key + account_number]

	orders = [order for order in orders if order.pair == pair]

	total_profit = sum([order.PnL for order in orders])

	barrier_dir = avg_pair_barrier_num[pair] / avg_pair_barrier_denom[pair]

	if (barrier_dir > 0) != (trade_dir) and abs(barrier_dir) > 50 and total_profit > 0:

		print "close pair", pair

		success = True
		for order in orders:
			if close_order(order, api_key) == False:
				success = False

		if success:
			cursor = cnx.cursor()
			query = ("DELETE FROM trade_setups where api_key='" + api_key + "' and pair= '" + pair + "'")
			cursor.execute(query)
			cnx.commit()
			cursor.close()

			closed_setup.add(api_key + "_" + pair)


cnx.close()


