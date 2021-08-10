import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from pytz import timezone
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
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko


import os
from bisect import bisect

import paramiko

import logging
import os
import enum

import matplotlib
matplotlib.use('Agg')

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from bisect import bisect
import psutil
import logging
import datetime as dt
from uuid import getnode as get_mac
import traceback
import bisect

currency_pairs = [
	"AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
	"AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
	"AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
	"AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
	"AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
	"CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
	"CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

if get_mac() != 150538578859218:
	root_dir = "/root/" 
else:
	root_dir = "/tmp/" 


class MyFormatter(logging.Formatter):
	converter=dt.datetime.fromtimestamp
	def formatTime(self, record, datefmt=None):
		ct = self.converter(record.created)
		if datefmt:
			s = ct.strftime(datefmt)
		else:
			t = ct.strftime("%Y-%m-%d %H:%M:%S")
			s = "%s,%03d" % (t, record.msecs)
		return s

formatter = MyFormatter(fmt='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""

	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger



def get_time_series(symbol, time, granularity="H1"):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=" + granularity + "&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['candles']

	prices = []
	times = []

	index = 0
	while index < len(j):
		item = j[index]

		s = item['time']
		s = s[0 : s.index('.')]
		timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

		if item['lowMid'] != item['highMid'] or item['volume'] > 0:
			times.append(timestamp)
			prices.append(item['closeMid'])
			index += 1

	return prices, times

def load_time_series(symbol, year, is_bid_file):

	if get_mac() == 150538578859218:
		prefix = '/Users/andrewstevens/Downloads/economic_calendar/'
	else:
		prefix = '/root/trading_data/'

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [f for f in listdir(prefix) if isfile(join(prefix, f))]

	pair = symbol[0:3] + symbol[4:7]

	for file in onlyfiles:

		if pair in file and 'Candlestick_1_Hour_BID' in file:
			break

	if pair not in file:
		return None

	with open(prefix + file) as f:
		content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	from_zone = tz.gettz('America/New_York')
	to_zone = tz.tzutc()

	prices = []
	times = []
	volumes = []

	content = content[1:]

	if year != None:
		start_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".1.1 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())
		end_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".12.31 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())

	for index in range(len(content)):

		toks = content[index].split(',')
		utc = datetime.datetime.strptime(toks[0], "%d.%m.%Y %H:%M:%S.%f")

		time = calendar.timegm(utc.timetuple())

		if year == None or (time >= start_time and time < end_time):

			high = float(toks[2])
			low = float(toks[3])
			o_price = float(toks[1])
			c_price = float(toks[4])
			volume = float(toks[5])

			if high != low or volume > 0:
				prices.append(c_price)
				times.append(time)
				volumes.append(volume)

	return prices, times, volumes

def find_historic_trends(currency_pair, before_prices):

	prices, times, volumes = load_time_series(currency_pair, None, None)

	trends = []
	for index in range(80 * 24, len(prices)):

		trends.append(abs(np.mean([prices[index] - prices[index-back_step] for back_step in range(24, 80 * 24, 24)])))

	trends = sorted(trends)

	delta = abs(np.mean([before_prices[-1] - before_prices[-back_step] for back_step in range(2, 80)]))

	index = bisect.bisect(trends, delta)
	percentile = int((float(index) / len(trends)) * 100)

	return percentile

def generate_price_movement_table(trend_map, before_prices, currency_pair, global_ranking):

	if currency_pair[4:7] == "JPY":
		pip_size = 0.01
	else:
		pip_size = 0.0001

	trend_indicator = find_historic_trends(currency_pair, before_prices)
	is_trend = trend_indicator > 50

	if is_trend:
		table_str = "<table border=1 width=75%><tr><th colspan=12><center>Recommend <b>Trend Following</b> Strategy</center></th></tr>"
	else:
		table_str = "<table border=1 width=75%><tr><th colspan=12><center>Recommend <b>Trend Reversal</b> Strategy</center></th></tr>"

	table_str += "<tr><th>Time Frame</th><th>BUY / SELL</th><th>Price Delta (%)</th><th colspan=3><font color='red'>High Risk</font></th><th colspan=3><font color='orange'>Medium Risk</font></th><th colspan=3><font color='green'>Low Risk</font></th></tr>"
	table_str += "<tr><th colspan=3></th><th>TP Pips</th><th>SL Pips</th><th>Amount</th><th>TP Pips</th><th>SL Pips</th><th>Amount</th><th>TP Pips</th><th>SL Pips</th><th>Amount</th></tr>"

	final_decision = {}
	final_decision["is_trend"] = is_trend
	final_decision["trend_indicator"] = trend_indicator
	is_trend = False

	buy_percentiles = []
	sell_percentiles = []

	sell_indexes = []
	buy_indexes = []
	time_frames = ['1 Day', '2 Days', '3 Days', '4 Days', '1 Week', '2 Weeks', '3 Weeks', '1 Month', '2 Months', '3 Months', '4 Months', 'Average']
	for time_index, time_frame in zip([1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 0], time_frames):

		if time_index == 0:
			if len(buy_percentiles) > len(sell_percentiles):
				percentile = int(np.mean(buy_percentiles))
				time_indexes = buy_indexes
			else:
				percentile = int(np.mean(sell_percentiles))
				time_indexes = sell_indexes

			if is_trend: 
				delta = 1 if len(buy_percentiles) > len(sell_percentiles) else -1
			else:
				delta = 1 if len(buy_percentiles) < len(sell_percentiles) else -1

			if delta < 0:
				final_decision["percentile"] = -percentile
			else:
				final_decision["percentile"] = percentile

		else:
			delta = (before_prices[-1] - before_prices[-time_index-1]) / pip_size
			index = bisect.bisect(trend_map[time_index], abs(delta))

			percentile = int((float(index) / len(trend_map[time_index])) * 100)

		if time_index != 0:
			table_str += "<tr><td>{}</td>".format(time_frame)
		else:
			table_str += "<tr><td><b>{}</b></td>".format(time_frame)

		global_ranking.append({
			"time_frame" : time_frame,
			"percentile" : percentile,
			"dir" : delta > 0,
			"pair" : currency_pair
			})

		if is_trend:
			if delta < 0:
				is_buy = False
				table_str += "<td><font color='red'>SELL</font></td>"
				table_str += "<td><font color='red'>{}%</font></td>".format(percentile)
			else:
				is_buy = True
				table_str += "<td><font color='green'>BUY</font></td>"
				table_str += "<td><font color='green'>{}%</font></td>".format(percentile)
		else:
			if delta < 0:
				is_buy = True
				table_str += "<td><font color='green'>BUY</font></td>"
				table_str += "<td><font color='red'>{}%</font></td>".format(percentile)
			else:
				is_buy = False
				table_str += "<td><font color='red'>SELL</font></td>"
				table_str += "<td><font color='green'>{}%</font></td>".format(percentile)


		if time_index != 0:
			if is_buy:
				buy_percentiles.append(percentile)
				buy_indexes.append(time_index)
			else:
				sell_percentiles.append(percentile)
				sell_indexes.append(time_index)
		else:
			final_decision["is_buy"] = is_buy


		for sl in [50, 75, 95]:

			if time_index == 0:
				trends = [trend_map[index] for index in time_indexes]
				trends = [item for sublist in trends for item in sublist]
				range_val = int(np.percentile(trends, sl) - abs(delta))
				delta = np.mean(trends)
				final_decision["TP"] = int(abs(delta))
				final_decision["SL"] = range_val
			else:
				range_val = int(np.percentile(trend_map[time_index], sl) - abs(delta))

			if range_val <= 0:
				table_str += "<td colspan=3></td>"
				continue

			if (delta < 0) == is_trend:
				table_str += "<td><font color='red'>{}</font></td>".format(int(abs(delta)))
				table_str += "<td><font color='green'>{}</font></td>".format(range_val)
			else:
				table_str += "<td><font color='green'>{}</font></td>".format(int(abs(delta)))
				table_str += "<td><font color='red'>{}</font></td>".format(range_val)

			amount = int(round(100 * (float(percentile) / range_val)))
			table_str += "<td class='pip_diff'>{}</td>".format(amount)

			if time_index == 0 and sl == 95:
				final_decision["amount"] = amount

		table_str += "</tr>"

	table_str += "</table>"

	return table_str, final_decision


def get_price_trends(currency_pair, global_ranking):

	before_prices, times = get_time_series(currency_pair, 800, "D")

	if currency_pair[4:7] == "JPY":
		pip_size = 0.01
	else:
		pip_size = 0.0001

	trend_map = {}
	for index in range(1, len(before_prices)):
		for time_index in [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80]:
			if time_index not in trend_map:
				trend_map[time_index] = []

			if index + time_index < len(before_prices):
				trend_map[time_index].append(abs((before_prices[-index] - before_prices[-time_index - 1 - index]) / pip_size))

	for time_index in [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80]:
		trend_map[time_index] = sorted(trend_map[time_index])

	table_str, final_decision = generate_price_movement_table(trend_map, before_prices, currency_pair, global_ranking)

	delta_map = {}
	percentile_map = {}
	for time_index in [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80]:
		delta_map[time_index] = (before_prices[-1] - before_prices[-time_index-1]) / pip_size
		index = bisect.bisect(trend_map[time_index], abs(delta_map[time_index]))
		percentile_map[time_index] = int((float(index) / len(trend_map[time_index])) * 100)


	return delta_map, percentile_map, table_str, final_decision

def get_global_ranking_table(global_ranking):

	global_ranking = sorted(global_ranking, key=lambda x : x["percentile"], reverse = True)

	table_str = "<table class='table-bordered table-striped' style='border: 1px solid black;width:75%;'>"
	for item in global_ranking:

		if item["dir"]:
			table_str += "<tr><td>{}</td><td>{}</td><td><font color='lime'>{}%</font></td><td><font color='lime'>BUY</font></td></tr>".format(item["pair"], item["time_frame"], item["percentile"])
		else:
			table_str += "<tr><td>{}</td><td>{}</td><td><font color='red'>{}%</font></td><td><font color='red'>SELL</font></td></tr>".format(item["pair"], item["time_frame"], item["percentile"])

	return table_str + "</table>"

def draw_currency_map(currency_pair, global_ranking):

	local_ranking = []
	for item in global_ranking: 
		if item["pair"][0:3] not in currency_pair and item["pair"][4:7] not in currency_pair:
			continue

		if item["time_frame"] != "Average":
			continue

		if item["dir"] == False:
			item["percentile"] = -item["percentile"]

		if item["pair"][0:3] == currency_pair[0:3] or item["pair"][4:7] == currency_pair[4:7]:
			local_ranking.append(item)
		else:
			local_ranking.append({
				"pair" : item["pair"][4:7] + "_" + item["pair"][0:3], 
				"time_frame" : item["time_frame"],
				"percentile" : -item["percentile"]
				})

	local_ranking = sorted(local_ranking, key = lambda x: x["percentile"], reverse=False)

	objects = [v["pair"] for v in local_ranking]
	colors = ["lime" if v["percentile"] > 0 else "red" for v in local_ranking]
	percentiles = [v["percentile"] for v in local_ranking]
	y_pos = np.arange(len(objects))


	plt.figure(figsize=(8,4))
	mid_line = y_pos[int(len(y_pos) * 0.5)]
	plt.axvline(0, color='black')
	plt.axhline(mid_line, color='black')

	plt.title("{} Comparison Movements".format(currency_pair))
	plt.barh(y_pos, percentiles, color = colors, align='center', alpha=0.5)

	mean_line = [y for y, v in zip(y_pos, local_ranking) if v["pair"] == currency_pair]

	if mean_line[0] == mid_line:
		plt.axhline(mean_line[0], color='orange', label = "Basket Compairson NEUTRAL")
	elif mean_line[0] > mid_line:
		plt.axhline(mean_line[0], color='lime', label = "Basket Compairson BUY")
	else:
		plt.axhline(mean_line[0], color='red', label = "Basket Compairson SELL")
	
	plt.yticks(y_pos, objects)
	plt.xlabel('Percentile Movement (%)')
	plt.legend()
	#pdf.savefig()
	#plt.show()
	plt.savefig("/var/www/html/images/{}_basket_movements.png".format(currency_pair))
	plt.close()

def generate_summary_prediction(pair, final_decision):

	cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

	query = ("""SELECT pair, model_group, model_type, forecast_dir as trade_dir, timestamp, time_frame, forecast_percentiles FROM signal_summary 
				where pair = '{}' 
			    and DATEDIFF(now(), timestamp) < 1.5
				group by pair, model_group, model_type, forecast_dir, timestamp
				order by timestamp desc
		""".format(pair))

	cursor = cnx.cursor()
	cursor.execute(query)

	net_dir = 0
	total_count = 0
	rows = [row for row in cursor]

	model_group_total_count = {"Economic" : 0, "Technical" : 0, "Basket" : 0}
	model_group_dir = {"Economic" : 0, "Technical" : 0, "Basket" : 0}

	for threshold_cutoff in range(2, 18, 2):
		found_set = set()
		for row in rows:
			pair = row[0]
			model_group = row[1]
			model_type = row[2]
			trade_dir = int(row[3])
			time_frame = row[5]

			if model_group not in model_group_dir:
				continue

			if model_group == "Economic" and model_type in ["Barrier", "Forecast"] and time_frame != 1:
				continue

			if model_group + model_type in found_set:
				continue

			forecast_percentiles = json.loads(row[6])
			if len(forecast_percentiles) == 0:
				continue

			left = np.percentile(forecast_percentiles, 50 + threshold_cutoff)
			right = np.percentile(forecast_percentiles, 50 - threshold_cutoff)

			if (left > 0) == (right > 0):
				found_set.add(model_group + model_type)

				if left > 0:
					net_dir += 1
					model_group_dir[model_group] += 1
				else:
					net_dir -= 1
					model_group_dir[model_group] -= 1

				total_count += 1
				model_group_total_count[model_group] += 1

	return  {
		"pair" : pair, 
		"is_buy" : final_decision["is_buy"],
		"Basket" : int(100 * float(model_group_dir["Basket"]) / max(1, model_group_total_count["Basket"])),
		"Technical" : int(100 * float(model_group_dir["Technical"]) / max(1, model_group_total_count["Technical"])),
		"Economic" : int(100 * float(model_group_dir["Economic"]) / max(1, model_group_total_count["Economic"])),
		"Summary" : int(100 * (float(net_dir) / max(1, total_count))),
		"is_trend" : final_decision["is_trend"],
		"trend_indicator" : final_decision["trend_indicator"],
		"percentile" : final_decision["percentile"],
		"SL" : final_decision["SL"],
		"TP" : final_decision["TP"],
		"amount" : final_decision["amount"],
	}


def generate_trade_summary(trade_decisions):

	table_str = """<table class='table-bordered table-striped' style='border: 1px solid black;width:75%;'>
	<tr><th colspan=9><center><b>Trade Recommendations</b></center></th></tr>
	<tr><th>Pair</th><th>Direction</th><th>Amount</th><th>Stop Loss</th><th>Take Profit</th><th>Market Type</th><th>Average Movement</th><th>Reward / Risk</th><th>Action</th></tr>
	"""

	for decision in trade_decisions:

		for is_trend in [False, True]:

			if is_trend:
				trend_weight = float(decision["trend_indicator"]) / 100
				if decision["percentile"] < 0:
					is_buy = False
				else:
					is_buy = True
			else:
				trend_weight = 1 - (float(decision["trend_indicator"]) / 100)
				if decision["percentile"] < 0:
					is_buy = True
				else:
					is_buy = False

			reward_risk = round(decision["reward_risk"], 2)

			if is_buy != (decision["Summary"] > 0):
				continue

			if abs(reward_risk) < 4:

				if is_trend == False and (reward_risk > 0) != (decision["Summary"] > 0):
					continue

				if abs(decision["Summary"]) < 25:
					continue

			if abs(reward_risk) < 2:
				continue

			percentile_color = 'lime'
			if decision["percentile"] < 0:
				percentile_color = 'red'

			reward_risk_str = "<font color='lime'>{}</font>".format(abs(reward_risk)) if reward_risk > 0 else "<font color='red'>{}</font>".format(abs(reward_risk)) 
			
			if is_trend == False and abs(reward_risk) > 4:
				close_str = "Close"
			else:
				close_str = "<b>Open</b>"

			table_str += """<tr>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				<td align=middle class='pip_diff'>{}</td>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				<td align=middle><font color='{}'>{}%</font></td>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				</tr>
				""".format(decision["pair"], 
					"<font color='lime'>BUY</font>" if is_buy else "<font color='red'>SELL</font>", 
					decision["amount"] * trend_weight, 
					decision["SL"], 
					decision["TP"],
					"Trending" if is_trend else "Reverting", 
					percentile_color, abs(decision["percentile"]), 
					reward_risk_str, 
					close_str
					)

	table_str += "</table>"

	return table_str

def generate_indicator_summary(indicator_summary):

	table_str = """<table class='table-bordered table-striped' style='border: 1px solid black;width:75%;'>
	<tr><th colspan=6><center><b>Forecast Breakdown</b></center></tr>
	<tr><th>Pair</th><th>Amount</th><th>Summary</th><th>Economic</th><th>Technical</th><th>Basket</th></tr>
	"""

	indicator_summary = sorted(indicator_summary, key=lambda x: abs(x["Summary"]), reverse=True)

	for decision in indicator_summary:


		pair_str = "<font color='lime'>{}</font>".format(decision["pair"]) if (decision["Summary"] > 0) else "<font color='red'>{}</font>".format(decision["pair"])
	
		table_str += """<tr>
			<td align=middle>{}</td>
			<td align=middle class='pip_diff'>{}</td>
			<td align=middle>{}</td>
			<td align=middle>{}</td>
			<td align=middle>{}</td>
			<td align=middle>{}</td>
			</tr>""".format(
				pair_str, 
				abs(decision["Summary"]),
				"<font color='lime'>{}%</font>".format(decision["Summary"]) if decision["Summary"] > 0 else "<font color='red'>{}%</font>".format(abs(decision["Summary"])),
				"<font color='lime'>{}%</font>".format(decision["Economic"]) if decision["Economic"] > 0 else "<font color='red'>{}%</font>".format(abs(decision["Economic"])), 
				"<font color='lime'>{}%</font>".format(decision["Technical"]) if decision["Technical"] > 0 else "<font color='red'>{}%</font>".format(abs(decision["Technical"])), 
				"<font color='lime'>{}%</font>".format(decision["Basket"]) if decision["Basket"] > 0 else "<font color='red'>{}%</font>".format(abs(decision["Basket"]))
				)

	table_str += "</table>"

	return table_str

def generate_straddle_summary(indicator_summary):

	table_str = """<table class='table-bordered table-striped' style='border: 1px solid black;width:75%;'>
	<tr><th colspan=6><center><b>Straddle Strategy Recommendations</b></center></tr>
	<tr><th>Pair</th><th>Market Type</th><th>Confidence</th><th>Amount</th><th>Trailing Stop</th></tr>
	"""

	indicator_summary = sorted(indicator_summary, key=lambda x: abs(x["Summary"]), reverse=True)

	for decision in indicator_summary:

		if decision["is_trend"] == False:
			continue

		trend_weight = float(decision["trend_indicator"]) / 100

		table_str += """<tr>
			<td align=middle>{}</td>
			<td align=middle>Trending</td>
			<td align=middle>{}%</td>
			<td align=middle class='pip_diff'>{}</td>
			<td align=middle>{} pips</td>
			</tr>""".format(
				decision["pair"], 
				abs(decision["Summary"]),
				abs(decision["Summary"]) * trend_weight,
				decision["TS"]
				)

	table_str += "<tr><td colspan=5><center>For a straddle strategy, you open a <font color='lime'>BUY</font> and <font color='red'>SELL</font> position for the specified amount, with the trailing stop suggested.</center></td></tr>"
	table_str += "</table>"

	return table_str

def draw_support_resistance_chart():
	cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

	query = ("""SELECT pair, forecast_percentiles FROM signal_summary 
					where model_group = 'Support And Resistance'
					order by timestamp desc
					limit 48
			""")

	cursor = cnx.cursor()
	cursor.execute(query)

	rows = [row for row in cursor]

	pair_summary_map = {}
	for pair in currency_pairs:

		table_str = """<table class='table-bordered table-striped' style='border: 1px solid black;width:75%;'>
		<tr><th colspan=7><center><b>Support And Resistance Levels</b></center></tr>
		<tr><th>Pair</th><th>Direction</th><th>Percentile Level</th><th>Amount</th><th>Stop Loss</th><th>Take Profit</th><th>Reward Risk Ratio</th></tr>
		"""
		
		first_row = [row for row in rows if row[0] == pair][0]
		metadata = json.loads(first_row[1])

		levels = metadata["levels"]
		curr_price = metadata["curr_price"]

		resistances = [level for level in levels if level > curr_price]
		supports = [level for level in levels if level < curr_price]


		resistances = sorted(resistances, reverse=False)
		supports = sorted(supports, reverse=True)

		if pair[4:7] == "JPY":
			pip_size = 0.01
			decimal_places = 2
		else:
			pip_size = 0.0001
			decimal_places = 4

		stop_losses = []
		reward_risks = []
		for barrier in range(5, 80, 5):

			if len(resistances) > 0:
				resistance = resistances[int(len(resistances) * barrier * 0.01)] - curr_price
				resistance_price = resistances[int(len(resistances) * barrier * 0.01)]
			else:
				resistance = None

			if len(supports) > 0:
				support = supports[int(len(supports) * barrier * 0.01)] - curr_price
				support_price = supports[int(len(supports) * barrier * 0.01)]
			else:
				support = None

			if support == None:
				support = resistance
				support_price = "NA"

			if resistance == None:
				resistance = support
				resistance_price = "NA"

			resistance /= pip_size
			support /= pip_size

			resistance = abs(resistance)
			support = abs(support)

			if resistance > support:
				is_buy = True
				sl = support
				sl_price = support_price
				tp = resistance
				tp_price = resistance_price
			else:
				is_buy = False
				sl = resistance
				sl_price = resistance_price
				tp = support
				tp_price = support_price

			if barrier < 50:
				if resistance > support:
					reward_risks.append(tp / sl)
				else:
					reward_risks.append(-tp / sl)

			amount = round(float(1000.0) / sl, 2)
			stop_losses.append(sl)

			table_str += """<tr>
			<td align=middle>{}</td>
			<td align=middle>{}</td>
			<td align=middle>{}%</td>
			<td align=middle class='pip_diff'>{}</td>
			<td align=middle>{} (<b>{}</b> pips)</td>
			<td align=middle>{} (<b>{}</b> pips)</td>
			<td align=middle>{}</td>
			</tr>""".format(
				pair, 
				"<font color='lime'>BUY</font>" if is_buy else "<font color='red'>SELL</font>",
				barrier,
				amount,
				round(sl_price, decimal_places),
				int(round(sl)),
				round(tp_price, decimal_places),
				int(round(tp)),
				round(tp / sl, 2)
				)



		table_str += "</table>"
		
		pair_summary_map[pair] = {
			"table" : table_str, 
			"TS" : int(np.mean(stop_losses)), 
			"reward_risk" : np.mean(reward_risks)}

	return pair_summary_map

def draw_pair_forecast_summary(decision):

	table_str = """<br><table class='table-bordered table-striped' style='border: 1px solid black;width:75%;'>
		<tr><th colspan=6><center><b>Forecast Breakdown</b></center></tr>
		<tr><th>Pair</th><th>Summary</th><th>Economic</th><th>Technical</th><th>Basket</th></tr>
		"""
	
	pair_str = "<font color='lime'>{}</font>".format(decision["pair"]) if (decision["Summary"] > 0) else "<font color='red'>{}</font>".format(decision["pair"])

	table_str += """<tr>
		<td align=middle>{}</td>
		<td align=middle>{}</td>
		<td align=middle>{}</td>
		<td align=middle>{}</td>
		<td align=middle>{}</td>
		</tr>""".format(
			pair_str, 
			"<font color='lime'>{}%</font>".format(decision["Summary"]) if decision["Summary"] > 0 else "<font color='red'>{}%</font>".format(abs(decision["Summary"])),
			"<font color='lime'>{}%</font>".format(decision["Economic"]) if decision["Economic"] > 0 else "<font color='red'>{}%</font>".format(abs(decision["Economic"])), 
			"<font color='lime'>{}%</font>".format(decision["Technical"]) if decision["Technical"] > 0 else "<font color='red'>{}%</font>".format(abs(decision["Technical"])), 
			"<font color='lime'>{}%</font>".format(decision["Basket"]) if decision["Basket"] > 0 else "<font color='red'>{}%</font>".format(abs(decision["Basket"]))
			)

	table_str += "</table>"


	return table_str

def get_today_prediction():

	support_resistance_map = draw_support_resistance_chart()

	pair_map = {}
	global_ranking = []
	indicator_summary = []
	for pair in currency_pairs:
		print (pair)

		delta_map, percentile_map, table_str, final_decision = get_price_trends(pair, global_ranking)
		

		indicator = generate_summary_prediction(pair, final_decision)

		indicator["reward_risk"] = support_resistance_map[pair]["reward_risk"]
		indicator["TS"] = support_resistance_map[pair]["TS"]
		indicator_summary.append(indicator)

		pair_map[pair] = {}
		pair_map[pair]["deltas"] = delta_map
		pair_map[pair]["percentiles"] = percentile_map
		pair_map[pair]["mov_table"] = support_resistance_map[pair]["table"] + draw_pair_forecast_summary(indicator)


	pair_map["global_ranking"] = get_global_ranking_table(global_ranking)

	pair_map["summary_table"] = generate_trade_summary(indicator_summary)
	pair_map["indicator_summary"] = generate_indicator_summary(indicator_summary)
	pair_map["straddle_summary"] = generate_straddle_summary(indicator_summary)

	pickle.dump(pair_map, open("{}price_deltas.pickle".format(root_dir), "wb"))
	pickle.dump(indicator_summary, open("{}trade_decisions.pickle".format(root_dir), "wb"))

	for pair in currency_pairs:
		draw_currency_map(pair, global_ranking)


trade_logger = setup_logger('first_logger', root_dir + "get_delta.log")
trade_logger.info('Starting ') 


try:
	get_today_prediction()
	trade_logger.info('Finished ') 
except:
	print (traceback.format_exc())
	trade_logger.info(traceback.format_exc())


