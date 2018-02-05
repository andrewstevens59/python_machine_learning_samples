import sys
import math
from datetime import datetime
from random import *
import os.path


import pickle

import pycurl
from StringIO import StringIO

import time
import datetime
import calendar
import json

import numpy as np


def find_avg_mov(symbol, time):
	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=D&dailyAlignment=0&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()

	import json
	j = json.loads(response_value)['candles']

	avg_mov = 0
	for item in j:
		avg_mov = avg_mov + abs(item['openMid'] - item['closeMid'])

	return avg_mov / len(j)

class Order:

    def __init__(self):
        self.pair = ""
        self.dir = 0
        self.open_price = 0
        self.time = 0
        self.readable_time = ""
        self.amount = 0
        self.id = 0
        self.side = 0
        self.pnl = 0
        self.open_price = 0
        self.tp_price = 0
        self.sl_price = 0

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

currencies = ['USD', 'EUR', 'NZD', 'AUD', 'CAD', 'GBP', 'JPY', 'CHF']

portfolio_map = {}
last_trade_time = {}
for currency in currencies:
	portfolio_map[currency] = 0
	last_trade_time[currency] = 0

avg_daily_mov = {}

def sendCurlRequest(url, request_type, post_data = None):
	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, url)

	curl.setopt(pycurl.CUSTOMREQUEST, request_type)

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	if post_data != None:
		print post_data
		curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8', 'Content-Type: application/json'])
		curl.setopt(pycurl.POSTFIELDS, post_data)
	else:
		curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()

	return response_value

def init_trades():

	exposure = {}
	for currency in currencies:
		exposure[currency] = 0

	shuffle(currency_pairs)

	for pair in currency_pairs:

		currency1 = pair[0:3]
		currency2 = pair[4:7]

		exposure1 = exposure[currency1]
		exposure2 = exposure[currency2]

		if abs(exposure1) > abs(exposure2):

			print sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/001-001-1370090-004/orders", "POST", '{ "order" : {"instrument": "' + pair + '", "units": 1, "type": "MARKET", "side" : "sell"}}')

			if exposure[currency1] > 0:
				exposure[currency1] = exposure[currency1] - 1
				exposure[currency2] = exposure[currency2] + 1
			else:
				exposure[currency1] = exposure[currency1] + 1
				exposure[currency2] = exposure[currency2] - 1
		else:

			print sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/001-001-1370090-004/orders", "POST", '{ "order" : {"instrument": "' + pair + '", "units": 1, "type": "MARKET", "side" : "buy"}}')

			if exposure[currency2] > 0:
				exposure[currency1] = exposure[currency1] + 1
				exposure[currency2] = exposure[currency2] - 1
			else:
				exposure[currency1] = exposure[currency1] - 1
				exposure[currency2] = exposure[currency2] + 1


count = 0
while True:
	response_value = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/001-001-1370090-004/", "GET")
	j = json.loads(response_value)
	profit = float(j['account'][u'unrealizedPL'])
	balance = float(j['account'][u'balance'])
	margin_used = float(j['account'][u'marginUsed'])

	print "Balance: ", balance, "Profit: ", profit, "Marin Used: ", margin_used

	time.sleep(60)

	print "Checking Open Trades"

	order_set = []

	max_order_size = {}
	min_price_order = {}
	max_price_order = {}

	response_value = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/001-001-1370090-004/trades", "GET")

	j = json.loads(response_value)
	open_orders = j['trades']

	currency_exposure = {}
	for currency in currencies:
		currency_exposure[currency] = 0


	net_trade_amount = {}
	net_trade_pnl = {}
	for pair in pairs:
		net_trade_amount[pair] = 0
		net_trade_pnl[pair] = 0

	exposure_sum = 0
	for trade in open_orders:

		s = trade['openTime'].replace(':', "-")
		s = s[0 : s.index('.')]
		open_time = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())

		order = Order()
		order.time = open_time
		order.id = trade['id']
		order.pair = trade['instrument'].replace("/", "_")
		order.readable_time = trade['openTime']
		order.amount = float(trade['currentUnits'])
		order.pnl = float(trade['unrealizedPL'])
		order.open_price = float(trade['price'])

		currency1 = trade['instrument'][0:3]
		currency2 = trade['instrument'][4:7]

		currency_exposure[currency1] = currency_exposure[currency1] + order.amount
		currency_exposure[currency2] = currency_exposure[currency2] - order.amount

		net_trade_amount[order.pair] = net_trade_amount[order.pair] + order.amount
		net_trade_pnl[order.pair] = net_trade_pnl[order.pair] + order.pnl

		exposure_sum = exposure_sum + order.amount

		if order.pair not in max_order_size:
			max_order_size[order.pair] = order
			min_price_order[order.pair] = order
			max_price_order[order.pair] = order

		if abs(order.amount) > abs(max_order_size[order.pair].amount):
			max_order_size[order.pair] = order

		if order.open_price > max_price_order[order.pair].open_price:
			max_price_order[order.pair] = order

		if order.open_price < min_price_order[order.pair].open_price:
			min_price_order[order.pair] = order

		last_trade_time[currency1] = max(last_trade_time[currency1], open_time)
		last_trade_time[currency2] = max(last_trade_time[currency2], open_time)

		order_set.append(order)

	order_set.sort(key = lambda x: x.time)
		

	if len(order_set) > 0:
		for order in order_set:
			exposure = (net_trade_amount[order.pair] / exposure_sum) * margin_used 
			if abs(net_trade_pnl[order.pair]) / exposure > 0.03:
				sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/001-001-1370090-004/trades/" + order.id + "/close", "PUT")



	exposure_sum = 0
	for currency in currencies:
		exposure_sum = exposure_sum + abs(currency_exposure[currency])

	avg_exposure = exposure_sum / len(currencies)

	curr_time = int(time.time())

	for pair in max_order_size:

		order = max_order_size[pair]

		num_trades = 3
		min_order_size = int(float(balance) / (6 * num_trades))

		time_diff = (curr_time - order.time) / (60 * 60)

		currency1 = pair[0:3]
		currency2 = pair[4:7]

		'''
		ratio1 = (abs(currency_exposure[currency1]) / avg_exposure)
		ratio2 = (abs(currency_exposure[currency2]) / avg_exposure)

		print ratio1, ratio2

		# if order in the same direction as net exposure than flip the ratio
		if (order.amount < 0) == (currency_exposure[currency1] < 0):
			ratio1 = 1 / ratio1

		if (order.amount > 0) == (currency_exposure[currency2] < 0):
			ratio2 = 1 / ratio2

		if abs(currency_exposure[currency1]) > abs(currency_exposure[currency2]):

			if abs(currency_exposure[currency1]) < avg_exposure and (order.amount < 0) == (currency_exposure[currency1] < 0) and ratio1 < 1:
				ratio1 = 1 / ratio1

			avg_ratio = ratio1
		else:

			if abs(currency_exposure[currency2]) < avg_exposure and (order.amount > 0) == (currency_exposure[currency2] < 0) and ratio2 < 1:
				ratio2 = 1 / ratio2

			avg_ratio = ratio2

		avg_ratio = min(avg_ratio, 2)
		'''


		if time_diff < (24 / num_trades) and abs(order.amount) >= min_order_size:
			continue

		response = sendCurlRequest("https://api-fxtrade.oanda.com/v1/prices?instruments=" + order.pair, "GET")
		response = json.loads(response)['prices']

		curr_price = 0
		for price in response:
			if price['instrument'] == pair:
				curr_price = (price['bid'] + price['ask']) / 2
				break

		if curr_price == 0:
			print "price not found"
			continue

		if pair not in avg_daily_mov:
			avg_daily_mov[pair] = find_avg_mov(pair, 20)

		# the average amount it should move between trades
		delta = (avg_daily_mov[pair] / num_trades)

		if (curr_price > min_price_order[pair].open_price - delta) and (curr_price < max_price_order[pair].open_price + delta)  and (abs(order.amount) >= min_order_size):
			print pair, "curr price: ", curr_price, "Min Price: ", min_price_order[pair].open_price - delta, "Max Price: ", max_price_order[pair].open_price + delta
			continue

		avg_daily_mov[pair] = find_avg_mov(pair, 20)

		# the amount it should move in 1 day
		pip_val = avg_daily_mov[pair]
		precision = '%.4f'

		if order.pair[4:7] == 'JPY':
			precision = '%.3f'

		if order.amount > 0:
			tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (1.05 * pip_val))) + '"}'
			sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price - (pip_val))) + '"}'
			print sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/001-001-1370090-004/orders", "POST", '{ "order" : {' + tp_price + ', ' + sl_price + ', "instrument": "' + order.pair + '", "units": ' + str(int(max(min_order_size, 0) * 1)) + ', "type": "MARKET", "side" : "buy"}}')
		else:
			tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price - (1.05 * pip_val))) + '"}'
			sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (pip_val))) + '"}'
			print sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/001-001-1370090-004/orders", "POST", '{ "order" : {' + tp_price + ', ' + sl_price + ', "instrument": "' + order.pair + '", "units": ' + str(-int(max(min_order_size, 0) * 1)) + ', "type": "MARKET", "side" : "sell"}}')

		

	for currency in currencies:
		currency_exposure[currency] = currency_exposure[currency] / max(1, exposure_sum)

	print currency_exposure

	if profit / balance > 0.01 or profit / margin_used > 0.02:
		print "Close All"
		
		for order in order_set:
			sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/001-001-1370090-004/trades/" + order.id + "/close", "PUT")


