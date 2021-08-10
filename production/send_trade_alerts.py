import sys
import math
from datetime import datetime
import random
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
import json
import requests

import os
from bisect import bisect

import paramiko
import json

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
	root_dir = "../" 


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


def get_today_prediction():

	cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')


	trade_decisions = pickle.load(open("{}trade_decisions.pickle".format(root_dir), "rb"))

	is_new_trade_alert = False
	for pair in currency_pairs:

		print (pair)

		query = ("""SELECT metadata  FROM trade_alerts 
				where pair = '{}'
				order by timestamp desc
					limit 1
			""".format(pair))

		cursor = cnx.cursor()
		cursor.execute(query)
		rows = [row for row in cursor]

		if len(rows) > 0:
			prev_trade_decision = json.loads(rows[0][0])
		else:
			prev_trade_decision = None

		for decision in trade_decisions:

			if decision["pair"] != pair:
				continue

			is_close = False
			if prev_trade_decision is not None:
				if prev_trade_decision["is_buy"] == decision["is_buy"]:
			
					if abs(prev_trade_decision["percentile"] - decision["percentile"]) < 25:
						continue
				else:
					is_close = True
			
			is_recommend = (decision["is_buy"] == (decision["Summary"] > 0)) and abs(decision["Summary"]) > 30


			if is_recommend or is_close:
				decision["is_recommend"] = is_recommend
				decision["is_close"] = is_close

				is_new_trade_alert = True
				cursor = cnx.cursor()
				query = ("""INSERT INTO trade_alerts(timestamp, pair, metadata) 
							values (now(),'{}','{}')""".format(
						pair,
						json.dumps(decision)
						))

				cursor.execute(query)
				cnx.commit()

	return is_new_trade_alert

def send_information_email(user):

	
	df = pd.read_csv("{}emails/foreign-realestate-investment-group.csv".format(root_dir))
	emails = df["email"]
	emails.dropna(inplace=True)
	emails = emails.values.tolist()

	emails = [email.replace("'", "") for email in emails]
	

	email_template = ""
	#email_template += '<img src="data:image/jpeg;base64,{}" alt="img" />'.format(encoded_string)
	import datetime
	email_template += """


	  	<h3> <font color='blue'>Expand Your Future With</font> <font color='red'>QuantForexTrading.com</font></h3>

        <img src="http://quantforextrading.com/images/ibize.png">
        <br>

        <p style="font-family:monospace; font-size: 20px; color: white;">
            <span style="border: solid 10px black; background-color: black; border-radius:5px;">
                "Want to live here?
                Trade Forex Profitably with QuantForexTrading.com"
            </span>
        </p>


	"""

	#letters = string.ascii_lowercase
	#email_template += "<div style=\"display:none;\">" + ''.join(random.choice(letters) for i in range(100)) + "</div>"


	print (email_template)

	for email in emails:
		print (email)
		requests.post("https://quantforextrading.com/html/send_email.php", data={'to': email, 'subject': 'QuantForexTrading - Free Forex Education Course ({})'.format(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")), 'text': email_template})


def send_email_template(cursor, cnx, user, existing_alerts, pair_row_map, is_marketing = False):

	alert_count = 0
	email_template = "<center><H2><font color='blue'>QuantForexTrading.com</font> Trade Alert Summary</H2></center>" 
	if user["upgrade_status"] == 0:
		email_template += "<H3><a href=\"https://quantforextrading.com/?view=my_account.php\"><font color='red'>Upgrade Now</font></a> To Get Most Recent Trade Alerts</H3>"
		email_template += "<b>Standard Members</b> limited to weekly reports only, meaning you will miss many new trades."

	email_template += "<table style='border: 1px solid black;width:100%;'><tr><th>Pair</th><th>Direction</th><th>Amount</th><th>Stop Loss</th><th>Take Profit</th><th>Market Type</th><th>Average Movement</th></tr>"
	for pair in currency_pairs:
		rows = pair_row_map[pair]
		
		if len(rows) == 0:
			continue

		if str(user["user_id"]) + str(rows[0][1]) + pair in existing_alerts:
			continue

		alert_count += 1
		curr_alert = json.loads(rows[0][0])

		order_size = curr_alert["amount"]
		if user["upgrade_status"] != 0:
			if user["account_size"] > 0:
				order_size = int(3 * curr_alert["amount"] * (user["account_size"] / 1500))
				
			query = ("""INSERT INTO user_trade_alerts_provided(timestamp, pair, user_id) 
							values ('{}','{}','{}')""".format(
						rows[0][1],
						pair,
						user["user_id"]
						))

			cursor.execute(query)
			cnx.commit()

		percentile_color = 'lime'
		if curr_alert["percentile"] < 0:
			percentile_color = 'red'

		if "is_recommend" not in curr_alert or curr_alert["is_recommend"]:
			email_template += """<tr>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				<td align=middle>{}</td>
				<td align=middle><font color='{}'>{}%</font></td></tr>""".format(curr_alert["pair"], 
					"<font color='lime'>BUY</font>" if curr_alert["is_buy"] else "<font color='red'>SELL</font>", 
					order_size, 
					curr_alert["SL"], 
					curr_alert["TP"],
					"Trending" if curr_alert["is_trend"] else "Reverting", 
					percentile_color, abs(curr_alert["percentile"]), 
					)
		elif "is_close" in curr_alert and curr_alert["is_close"]:
			email_template += """<tr>
				<td align=middle>{}</td>
				<td align=middle>Close</td>
				<td align=middle colspan=3></td>
				<td align=middle>{}</td>
				<td align=middle><font color='{}'>{}%</font></td></tr>""".format(curr_alert["pair"], 
					"Trending" if curr_alert["is_trend"] else "Reverting", 
					percentile_color, abs(curr_alert["percentile"]), 
					)

	if alert_count > 0:

		if is_marketing:	
			trade_logger.info('Marketing Email ' + str(user["email"])) 
			print (user["email"])
		else:
			trade_logger.info('Sent Email To ' + str(user["email"])) 

		email_template += "</table>"
		email_template += "Check the <a href=\"https://quantforextrading.com/\">website</a> for the most recent detailed forecast charts for each pair. Automatically adjust your position size by entering you account size in the <b>My Account</b> page.</br>"

		email_template += "<h3><font color='blue'>QuantForexTrading</font> provides a statistical edge, to improve your risk adjusted return.</h3>"
		email_template += "<ul><li><b>Close An Open Position</b> if you see a trade recommendation in the opposite direction.</li>"
		email_template += "<li><b>Add To An Open Position</b> if you see a new trade recommendation in the same direction.</li></ul>"

		letters = string.ascii_lowercase
		email_template += "<div style=\"display:none;\">" + ''.join(random.choice(letters) for i in range(10000)) + "</div>"

		email_template += "<h3>Trade Recommendations Are Generated By</h3>"
		email_template += "<ul>"
		email_template += "<li>Training Large Numbers of Machine Learning Models On Historic Data</li>"
		email_template += "<li>Generating Large Number Of Predictions From Machine Learning Models</li>"
		email_template += "<li>Weighting Predictions By Confidence</li>"
		email_template += "<li>Combining Predictions Grouped By Economic, Technical and Basket Forecasts</li>"
		email_template += "</ul>"

		email_template += "<p>Best Regards,<br>The <b>QuantForexTrading</b> Team"

		email_template += "<br><br><a href=\"https://quantforextrading.com/html/unsubscribe.php?email={}\">Unsubscribe</a> from trade alerts".format(user["email"]) 

		import datetime

		requests.post("https://quantforextrading.com/html/send_email.php", data={'to': user["email"], 'subject': 'QuantForexTrading - Trade Recommendations Report ({})'.format(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")), 'text': email_template})

def send_marketing_emails():

	cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

	pair_row_map = {}
	for pair in currency_pairs:

		query = ("""SELECT metadata, timestamp  FROM trade_alerts 
					where pair = '{}'
					and DATEDIFF(timestamp, now()) < 1
					order by timestamp desc
						limit 2
				""".format(pair))

		cursor = cnx.cursor()
		cursor.execute(query)
		pair_row_map[pair] = [row for row in cursor]

	query = ("""SELECT email FROM marketing_emails
						where DATEDIFF(now(), last_trade_alert_email) < 30
						or unsubscribe = 1
						""")

	cursor = cnx.cursor()
	cursor.execute(query)
	rows = [row for row in cursor]
	existing_sent_emails = [row[0] for row in rows]

	query = ("""SELECT email FROM user
						""")

	cursor = cnx.cursor()
	cursor.execute(query)
	rows = [row for row in cursor]
	existing_sent_emails += [row[0] for row in rows]

	df = pd.read_csv("{}emails/foreign-realestate-investment-group.csv".format(root_dir))
	emails = df["email"]
	emails.dropna(inplace=True)
	emails = emails.values.tolist()

	df = pd.read_excel("{}emails/EzTrader.xls".format(root_dir))
	emails1 = df["Primary Email"]
	emails1.dropna(inplace=True)

	for email in emails1.values.tolist():

		try:
			emails.append(str(email))
		except:
			pass

	emails = [email.replace("'", "") for email in emails]

	rand.shuffle(emails)

	updated_emails = []
	for email in emails:
		
		if email in existing_sent_emails:
			continue

		updated_emails.append("'" + email + "'")
		try:
			query = ("""INSERT INTO marketing_emails(last_trade_alert_email, email) 
							values (now(),'{}')""".format(
						email
						))

			cursor.execute(query)
			cnx.commit()
		except:
			pass

		user = {"user_id" : 0, "email" : email, "upgrade_status" : 0}
		send_email_template(None, None, user, set(), pair_row_map, is_marketing = True)

		if len(updated_emails) > 100:
			break

	if len(updated_emails) > 0:
		query = ("""UPDATE marketing_emails SET last_trade_alert_email = now()
							where email in ({})
							""".format(",".join(updated_emails)))

		cursor = cnx.cursor()
		cursor.execute(query)



def generate_emails():

	cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

	query = ("""SELECT user_id, timestamp, pair FROM user_trade_alerts_provided
						where DATEDIFF(timestamp, now()) < 7
						""")

	cursor = cnx.cursor()
	cursor.execute(query)
	rows = [row for row in cursor]
	existing_alerts = set([str(row[0]) + str(row[1]) + row[2] for row in rows])


	query = ("""SELECT id as user_id, email, upgrade_status, account_size  FROM user 
				where unsubscribe = 0 and (DATEDIFF(now(), last_trade_alert_email) > 7
				or upgrade_status != 0
                or DATEDIFF(now(), last_trade_alert_email) is null)
			""")

	cursor = cnx.cursor()
	cursor.execute(query)
	rows = [row for row in cursor]
	email_user_ids = [{"user_id" : row[0], "email" : row[1], "upgrade_status" : row[2], "account_size" : row[3]} for row in rows]

	trade_decisions = pickle.load(open("{}trade_decisions.pickle".format(root_dir), "rb"))

	pair_row_map = {}
	for pair in currency_pairs:

		query = ("""SELECT metadata, timestamp  FROM trade_alerts 
					where pair = '{}'
					and DATEDIFF(timestamp, now()) < 1
					order by timestamp desc
						limit 2
				""".format(pair))

		cursor = cnx.cursor()
		cursor.execute(query)
		pair_row_map[pair] = [row for row in cursor]

	select_user_ids = []
	for user in email_user_ids:
		select_user_ids.append(str(user["user_id"]))

		print (user["email"])

		send_email_template(cursor, cnx, user, existing_alerts, pair_row_map)

		
	query = ("""UPDATE user SET last_trade_alert_email = now()
						where id in ({})
						""".format(",".join(select_user_ids)))

	cursor = cnx.cursor()
	cursor.execute(query)

	return pair_row_map

'''
Hello QuantForexTrading Members,

We are pleased to announce we currently able to offer automated trading through the platform. We offer our new automated statistical arbitrage strategy, with excellent historical performance.

View our Automated Trading Page (see attached).

https://quantforextrading.com/?view=automated_trading.html

If you would like to try the strategy on demo first, please get in contact and we can set it up for you.

Best Regards,
QuantForexTrading Team
'''

def get_email_list_meetup():
	df = pd.read_csv("/Users/andrewstevens/Downloads/foreign-realestate-investment-group_members_1614994970.csv")
	print (df)
	emails = df["email"]
	emails.dropna(inplace=True)
	emails = emails.values.tolist()
	for i in range(0, len(emails), 40):
		print (",".join(emails[i:i+40]))
	print (len(emails))
	sys.exit(0)

get_email_list_meetup()

trade_logger = setup_logger('first_logger', root_dir + "email_trade_alerts.log")
trade_logger.info('Starting ') 

print ("here", len(sys.argv))
if len(sys.argv) > 1 and sys.argv[1] == "test":
	print ("sending test email")
	cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

	pair_row_map = {}
	for pair in currency_pairs:

		query = ("""SELECT metadata, timestamp  FROM trade_alerts 
					where pair = '{}'
					and DATEDIFF(timestamp, now()) < 1
					order by timestamp desc
						limit 2
				""".format(pair))

		cursor = cnx.cursor()
		cursor.execute(query)
		pair_row_map[pair] = [row for row in cursor]

	user = {"email" : "andrew.stevens.591@gmail.com", "user_id" : 0, "upgrade_status" : 0}
	#send_email_template(None, None, user, set(), pair_row_map, is_marketing = False)
	send_information_email(user)
	sys.exit(0)



try:
	send_marketing_emails()
	is_new_trade_alert = get_today_prediction()
	if is_new_trade_alert:
		generate_emails()

	trade_logger.info('Finished ') 
except:
	print (traceback.format_exc())
	trade_logger.info(traceback.format_exc())


