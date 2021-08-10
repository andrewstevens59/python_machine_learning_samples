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
from bisect import bisect

import socket
import sys
import time

import math
import sys
import re

import os
import mysql.connector
import traceback


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
import os
import datetime as dt
import io



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


def load_model_distributions(cnx, pair, model_types):

    dist_map = {}
    dist_map["auc"] = {"All" : []}
    dist_map["rmse"] = {"All" : []}

    dist_map["prob"] = {"All" : []}
    dist_map["forecast"] = {"All" : []}

    query = ("""
        SELECT auc, rmse, barrier, model_type, prob - 0.5, forecast  
        FROM model_historical_stats 
        where pair = '{}'
        and model_type in ({})
        """.format(pair, model_types))

    cursor = cnx.cursor()
    cursor.execute(query)

    for row1 in cursor:
        auc = row1[0]
        rmse = row1[1]
        barrier = int(row1[2] / 24 * 5)
        model_type = row1[3]
        prob = row1[4]
        forecast = row1[5]

        if auc != 0:
            dist_map["auc"]["All"].append(auc)

            if model_type not in dist_map["prob"]:
                dist_map["prob"][model_type] = []

            dist_map["prob"][model_type].append(prob)
            dist_map["prob"]["All"].append(prob)

        if rmse != 0:
            dist_map["rmse"]["All"].append(rmse)
            dist_map["forecast"]["All"].append(forecast)

    dist_map["auc"]["All"] = sorted(dist_map["auc"]["All"])
    dist_map["rmse"]["All"] = sorted(dist_map["rmse"]["All"])

    return dist_map

def process_pending_trades_barrier(dist_map, select_pair, signals_file_prefix, max_barrier, select_auc_mult):


    auc_barrier_mult = select_auc_mult

    base_calendar = pickle.load(open(signals_file_prefix + select_pair + ".pickle"))
    description_forecast = {}
  
    prob_zero_perc = bisect(dist_map["prob"]["All"], 0)
    forecast_zero_perc = bisect(dist_map["forecast"]["All"], 0)

    if select_pair in base_calendar:
        for prediction in base_calendar[select_pair]:

            if prediction["auc"] < 0.45 or prediction["sharpe"] < 0:
                continue

            threshold_cutoff = max(0.05, 1.0 - ((select_auc_mult - 0.5) / (2.8 - 0.5)))

            metric = prediction["probability"] - 0.5
            prob_percentile = (float(bisect(dist_map["prob"]["All"], metric) - prob_zero_perc) / len(dist_map["prob"]["All"])) 
            auc_percentile = 1.0 - (min(0.15, max(0, prediction["auc"] - 0.5)) / 0.15)
            ratio = (prob_percentile / (1 + auc_percentile))


            if abs(ratio) >= threshold_cutoff:
                description = prediction["description"] + "_" + prediction["currency"] + "_auc"

                if description not in description_forecast:
                    description_forecast[description] = []

                description_forecast[description].append(ratio)

            metric = prediction["forecast"]
            forecast_percentile = float(bisect(dist_map["forecast"]["All"], metric) - forecast_zero_perc) / len(dist_map["forecast"]["All"])
            rmse_percentile = (float(bisect(dist_map["rmse"]["All"], prediction["rmse"])) / len(dist_map["rmse"]["All"]))
            ratio = (forecast_percentile / (1 + rmse_percentile))

            if abs(ratio) >= threshold_cutoff:
                description = prediction["description"] + "_" + prediction["currency"] + "_rmse"

                if description not in description_forecast:
                    description_forecast[description] = []

                description_forecast[description].append(ratio)

    
    signals = []
    if len(description_forecast) > 0:
        for description in description_forecast:
            signals.append(np.percentile(description_forecast[description], 25))
            signals.append(np.percentile(description_forecast[description], 50))
            signals.append(np.percentile(description_forecast[description], 75))

        return signals, base_calendar["price_trends"]

    return signals, base_calendar["price_trends"]

def process_pending_trades_regression(dist_map, select_pair, signals_file_prefix, max_barrier, select_auc_mult):

    base_calendar = pickle.load(open(signals_file_prefix + select_pair + ".pickle"))
   

    description_forecast = {}

    if select_pair in base_calendar:
        for prediction in base_calendar[select_pair]:

            if prediction["sharpe"] < 0:
                continue

            metric = prediction["forecast"]
            forecast_zero_perc = bisect(dist_map["forecast"]["All"], 0)
            forecast_percentile = float(bisect(dist_map["forecast"]["All"], metric) - forecast_zero_perc) / len(dist_map["forecast"]["All"])
            rmse_percentile = (float(bisect(dist_map["rmse"]["All"], prediction["rmse"])) / len(dist_map["rmse"]["All"]))

            ratio = (forecast_percentile / (1 + rmse_percentile))
            threshold_cutoff = max(0.05, 1.0 - ((select_auc_mult - 0.5) / (2.8 - 0.5)))

            print (ratio, threshold_cutoff, "********")

            if abs(ratio) < threshold_cutoff:
                continue

            description = prediction["description"] + "_" + prediction["currency"]
            if description not in description_forecast:
                description_forecast[description] = []

            description_forecast[description].append(ratio)
 
    signals = []
    if len(description_forecast) > 0:
        for description in description_forecast:
            signals.append(np.percentile(description_forecast[description], 25))
            signals.append(np.percentile(description_forecast[description], 50))
            signals.append(np.percentile(description_forecast[description], 75))
            
        return signals, base_calendar["price_trends"]

    return signals, base_calendar["price_trends"]

def process_pending_trades_regression_test(dist_map, select_pair, signals_file_prefix, max_barrier, select_auc_mult):

    base_calendar = pickle.load(open(signals_file_prefix + select_pair + ".pickle"))
   

    description_forecast = {}

    if select_pair in base_calendar:
        for prediction in base_calendar[select_pair]:

            if prediction["sharpe"] < 0:
                continue

            metric = prediction["forecast"]
            forecast_zero_perc = bisect(dist_map["forecast"]["All"], 0)
            forecast_percentile = float(bisect(dist_map["forecast"]["All"], metric) - forecast_zero_perc) / len(dist_map["forecast"]["All"])
            rmse_percentile = (float(bisect(dist_map["rmse"]["All"], prediction["rmse"])) / len(dist_map["rmse"]["All"]))

            ratio = (forecast_percentile / (1 + rmse_percentile))
            threshold_cutoff = 1.0 - ((select_auc_mult - 0.4) / (2.8 - 0.4))

            print (ratio, threshold_cutoff, "********")

            if abs(ratio) < threshold_cutoff:
                continue

def store_regression_predictions(cnx, select_pair, signals_file_prefix, model_type, last_signal_update_time):


    base_calendar = pickle.load(open(signals_file_prefix + select_pair + ".pickle"))
    cursor = cnx.cursor()

    if select_pair in base_calendar:
        for prediction in base_calendar[select_pair]:


            if "TT" in prediction:
                TT = prediction["TT"]
            else:
                TT = 0

            if "FF" in prediction:
                FF = prediction["FF"]
            else:
                FF = 0

            if "FT" in prediction:
                FT = prediction["FT"]
            else:
                FT = 0

            if "TF" in prediction:
                TF = prediction["TF"]
            else:
                TF = 0

            query = ("""INSERT INTO model_historical_stats(pair, model_type, barrier, forecast, rmse, r_2, sharpe, description, currency, timestamp, TT, FF, TF, FT) 
                values ('{}','{}','{}','{}','{}','{}','{}','{}', '{}', '{}','{}','{}', '{}', '{}')""".
                format(
                    select_pair,
                    model_type,
                    prediction["time_wait"],
                    prediction["forecast"],
                    prediction["rmse"],
                    prediction["r_2"],
                    prediction["sharpe"],
                    prediction["description"],
                    prediction["currency"],
                    last_signal_update_time,
                    TT,
                    FF,
                    TF,
                    FT
                    ))

            print (query)

            cursor.execute(query)
            cnx.commit()

    cursor.close()

def store_classification_predictions(cnx, select_pair, signals_file_prefix, model_type, last_signal_update_time):

    base_calendar = pickle.load(open(signals_file_prefix + select_pair + ".pickle"))
    cursor = cnx.cursor()

    if select_pair in base_calendar:
        for prediction in base_calendar[select_pair]:

            if "TT" in prediction:
                TT = prediction["TT"]
            else:
                TT = 0

            if "FF" in prediction:
                FF = prediction["FF"]
            else:
                FF = 0

            if "FT" in prediction:
                FT = prediction["FT"]
            else:
                FT = 0

            if "TF" in prediction:
                TF = prediction["TF"]
            else:
                TF = 0

            query = ("""INSERT INTO model_historical_stats(pair, model_type, barrier, prob, auc, sharpe, forecast, rmse, description, currency, timestamp, TT, FF, TF, FT) 
                values ('{}','{}','{}','{}','{}','{}','{}','{}', '{}','{}', '{}','{}', '{}','{}', '{}')""".
                format(
                    select_pair,
                    model_type,
                    prediction["barrier"],
                    prediction["probability"],
                    prediction["auc"],
                    prediction["sharpe"],
                    prediction["forecast"],
                    prediction["rmse"],
                    prediction["description"],
                    prediction["currency"],
                    last_signal_update_time,
                    TT,
                    FF,
                    TF,
                    FT
                    ))

            print (query)

            cursor.execute(query)
            cnx.commit()

    cursor.close()

def store_live_history(pair, cnx, news_file, model_type, is_barrier):

    last_signal_update_time = os.path.getmtime(news_file + pair + ".pickle")
    print ("update time", (time.time() - last_signal_update_time) / (60 * 60), last_signal_update_time)
 
    if os.path.isfile(root_dir + "live_news_update_file.pickle"):
        with open(root_dir + "live_news_update_file.pickle", "rb") as f:
            live_news_update= pickle.load(f)
    else:
        live_news_update = {}

    key = pair + news_file
    if key in live_news_update:
        print ("last_signal_update_time", key, live_news_update[key], last_signal_update_time)
        if live_news_update[key] == last_signal_update_time:
            time_pip_diff = pickle.load(open(news_file + pair + ".pickle"))["price_trends"]
            return time_pip_diff

    barrier_models = ",".join(["'M1'", "'M9'", "'M15'", "'M21'"])
    regression_models = ",".join(["'M3'", "'M6'", "'M12'", "'M18'"])

    if is_barrier:
        dist_map = load_model_distributions(cnx, pair, barrier_models)
        store_classification_predictions(cnx, pair, news_file, model_type, last_signal_update_time)
    else:
        dist_map = load_model_distributions(cnx, pair, regression_models)
        store_regression_predictions(cnx, pair, news_file, model_type, last_signal_update_time)


    for auc_mult in np.arange(0.5,2.8,0.1):

        auc_mult_round = round(auc_mult, 1)
        if is_barrier:
            signals, time_pip_diff = process_pending_trades_barrier(dist_map, pair, news_file, 100, select_auc_mult = auc_mult_round)
        else:
            signals, time_pip_diff = process_pending_trades_regression(dist_map, pair, news_file, 100, auc_mult_round)

        cursor = cnx.cursor()
        for signal in signals:
            try:
                query = ("INSERT INTO live_signal_history values ( \
                    '" + pair + "', \
                    '" + str(signal) + "', \
                    '" + str(auc_mult_round) + "', \
                    '" + str(last_signal_update_time) + "', \
                    '" + str(model_type) + "'\
                    )")

                print (query)

                cursor.execute(query)
                cnx.commit()
            except:
                pass

        cursor.close()


    live_news_update[key] = last_signal_update_time

    with open(root_dir + "live_news_update_file.pickle", "wb") as f:
        pickle.dump(live_news_update, f)

    return time_pip_diff

import psutil

def checkIfProcessRunning(processName, command):
    count = 0
    #Iterate over the all the running process
    for proc in psutil.process_iter():

        try:
            cmdline = proc.cmdline()

            # Check if process name contains the given name string.
            if len(cmdline) > 2 and processName.lower() in cmdline[1] and command == cmdline[2]: 
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if count >= 2:
        sys.exit(0)

checkIfProcessRunning('execute_news_signals_summary.py', sys.argv[1])


def is_valid_trading_period(ts):

    date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    if s.weekday() in {4} and s.hour in {21, 22, 23}:
        return False
    if s.weekday() in {5}:
        return False
    if s.weekday() in {6} and s.hour < 21:
        return False
    
    return True

'''
if is_valid_trading_period(time.time()) == False:
    sys.exit(0)
'''

cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')


def generate_rows(time_frame, price_trends_diff, count):
    pair_rows = []
    probabilities = []
    pairs = []

    
    for pair in currency_pairs:

        rows = []
        '''
        rows.append("<table class='table-bordered table-striped' style='border: 1px solid black;width:100%;'>")

        if time_frame == 30:
            rows.append("<tr><td align=middle>Pair</td><td>Model</td><td align=middle>Monthly</td><td>Pip Diff</td><td>Percentile</td><td>Amount</td><td colspan=45 align=middle>High Confidence (%) <-----------------> Low Confidence (%)</td></tr>")
        elif time_frame == 7:
            rows.append("<tr><td align=middle>Pair</td><td>Model</td><td align=middle>Weekly</td><td>Pip Diff</td><td>Percentile</td><td>Amount</td><td colspan=45 align=middle>High Confidence (%) <-----------------> Low Confidence (%)</td></tr>")
        else:
            rows.append("<tr><td align=middle>Pair</td><td>Model</td><td align=middle>Daily</td><td>Pip Diff</td><td>Percentile</td><td>Amount</td><td colspan=45 align=middle>High Confidence (%) <-----------------> Low Confidence (%)</td></tr>") 


        all_pip_diffs = [] 
        all_signals = []

        for model_type in ["M1", "M3", "M6", "M9", "M12", "M15", "M18", "M21", "All"]:

            if model_type == "All":
                rows.append("<tr><td colspan=45><CENTER><B>Summary</B></CENTER></td></tr>")

            for week in range(1, 5):

                if model_type != "All":
                    show_model = model_type
                else:
                    show_model = "<B>" + model_type + "</B>"

                if time_frame == 30:
                    pip_diff = int(price_trends_diff[pair]["deltas"][week * 20])
                    percentile = int(price_trends_diff[pair]["percentiles"][week * 20])
                    period = "Month"
                elif time_frame == 7:
                    pip_diff = int(price_trends_diff[pair]["deltas"][week * 5])
                    percentile = int(price_trends_diff[pair]["percentiles"][week * 5])
                    period = "Week"
                else:
                    pip_diff = int(price_trends_diff[pair]["deltas"][week * 1])
                    percentile = int(price_trends_diff[pair]["percentiles"][week * 1])
                    period = "Day"

                color = ("green" if pip_diff > 0 else "red")

                line = """<tr>
                    <td>{}</td>
                    <td>{}</td>
                    <td aign=middle>{} {}</td>
                    <td style='color:{}'>{}</td>
                    <td style='color:{}'>{}%</td>
                    <td class='pip_diff'>{}</td>""".format(pair, show_model, week, period, color, pip_diff, color, percentile, abs(percentile))


                if model_type == "All":
                    all_pip_diffs.append(pip_diff)

                if model_type != "All":
                    query = ("SELECT avg(avg_signal), sum((case when avg_signal > 0 then 1 WHEN avg_signal < 0 THEN -1 else 0 end)) / count(*), auc_mult FROM live_signal_history where avg_signal != 0 and \
                                                pair = '" + pair + "' and model_type='" + str(model_type) + "' and ROUND((" + str(time.time()) + " - timestamp) / (60 * 60 * 24 * " + str(time_frame) + "), 0) < " + str(week) + " group by auc_mult order by auc_mult ")
                else:
                    query = ("SELECT avg(avg_signal), sum((case when avg_signal > 0 then 1 WHEN avg_signal < 0 THEN -1 else 0 end)) / count(*), auc_mult FROM live_signal_history where avg_signal != 0 and \
                                                pair = '" + pair + "' and ROUND((" + str(time.time()) + " - timestamp) / (60 * 60 * 24 * " + str(time_frame) + "), 0) < " + str(week) + " group by auc_mult order by auc_mult ")

                cursor = cnx.cursor()
                cursor.execute(query)

                data_outputs = []
                for row1 in cursor:
                    data_outputs.append(row1)

                for auc_mult in np.arange(0.5,2.8,0.1):

                    found = False
                    for row1 in data_outputs:
                        avg_signal = row1[0]
                        prob = int(round(row1[1] * 100))
                        auc_mult_compare = row1[2]

                        if auc_mult_compare == round(auc_mult, 2):
                            if model_type == "All":
                                all_signals.append(prob)

                            probabilities.append(prob)
                            pairs.append(pair)
                            line += ("?")
                            found = True
                            break

                    if found == False:
                        probabilities.append(0)
                        pairs.append(pair)
                        line += ("?")


                line += ("</tr>")
                rows.append(line)

            rows.append("<tr><td colspan=45></td></tr>")

        rows.append("</table>")
        '''

        img1 = "{}_barrier_{}_all_{}_time_frame_{}.png".format(pair, False, False, time_frame)
        img2 = "{}_barrier_{}_all_{}_time_frame_{}.png".format(pair, False, True, time_frame)
        img3 = "{}_barrier_{}_all_{}_time_frame_{}.png".format(pair, True, False, time_frame)
        img4 = "{}_barrier_{}_all_{}_time_frame_{}.png".format(pair, True, True, time_frame)
        img5 = "{}_error_percentage_time_frame_{}.png".format(pair, time_frame)
        img6 = "{}_regression_economic_forecast.png".format(pair)
        img7 = "{}_technical_forecast.png".format(pair)
        img8 = "{}_technical_barrier.png".format(pair)
        img9 = "{}_technical_exceed.png".format(pair)
        img10 = "{}_1_day_economic_forecast.png".format(pair)
        img11 = "{}_percentile_movements.png".format(pair)
        img12 = "{}_basket_forecast.png".format(pair)
        img13 = "{}_basket_barrier.png".format(pair)
        img14 = "{}_basket_movements.png".format(pair)
        img15 = "{}_support_resistance.png".format(pair)
        img16 = "{}_support_resistance_distribution.png".format(pair)
        img18 = "{}_trade_entries.png".format(pair)

        img19 = "{}_related_pairs.png".format(pair[0:3])
        img20 = "{}_news_correlation.png".format(pair[0:3])
        img21 = "{}_related_pairs.png".format(pair[4:7])
        img22 = "{}_news_correlation.png".format(pair[4:7])
  

        if time_frame == 30:
            forecast_tag = " Monthly Forecast"
        elif time_frame == 7:
            forecast_tag = " Weekly Forecast"
        else:
            forecast_tag = " Daily Forecast"

        '''
        mean_pip_diff = np.mean(all_pip_diffs)
        mean_signal = np.mean(all_signals)

        if (mean_pip_diff > 0) != (mean_signal > 0):
            if mean_pip_diff > 0:
                forecast_tag += " <font color='red'>SELL</font>"
            else:
                forecast_tag += " <font color='green'>BUY</font>"
        '''

        key = pair + str(time_frame)
        mov_table = price_trends_diff[pair]["mov_table"]
        pair_rows.append("""
         <div class="panel panel-default" id="{}">
            <div class="panel-heading" role="tab" id="heading_{}">
                <a id="link_{}" class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion_3" aria-expanded="false">
                    <div class="icon-ac-wrap pr-20"><span class="plus-ac"><i class="fa fa-plus"></i></span><span class="minus-ac"><i class="fa fa-minus"></i></span></div>
                    <B>{}</B> {} 
                </a>
            </div>
            <div id="collapse_{}" class="panel-collapse collapse" role="tabpanel">
                <div class="panel-body pa-15"> 

                <center>{}</center>
                <center>Recommended To Only Enter A Trade In The Direction Of The Forecast.</center>
                <center>{}</center>
                <center>news_summary_{}</center>
                

                <center><font color='red'>Use <b>Firefox Browser</b> If Images Do Not Display</font></center>
                <div id="content_{}"></div> 
                </div>
            </div>
        </div>

        <script>
        $('#link_{}').click(function(){{
            $('#content_{}').html(`
                <table border=0 style="width:100%;">
                <tr><td colspan=2><center><img src="http://104.237.4.171/images/{}" /></center></td></tr>
                <tr><td colspan=2><center><img src="http://104.237.4.171/images/{}" /></center></td></tr>
                <tr><td colspan=2><center><img src="http://104.237.4.171/images/{}" /></center></td></tr>
                <tr><td><img src="http://104.237.4.171/images/{}" /></td><td><img src="http://104.237.4.171/images/{}" /></td></tr>
                <tr><td><img src="http://104.237.4.171/images/{}" /></td><td><img src="http://104.237.4.171/images/{}" /></td></tr>
                <tr><td><img src="http://104.237.4.141/images/{}" /></td><td><img src="http://104.237.4.141/images/{}" /></td></tr>
                <tr><td><img src="http://104.237.4.141/images/{}" /></td><td><img src="http://104.237.4.141/images/{}" /></td></tr>
                <tr><td><img src="http://104.237.4.141/images/{}" /></td><td><img src="http://104.237.4.35/images/{}" /></td></tr>
                <tr><td><img src="http://104.237.4.174/images/{}" /></td><td><img src="http://104.237.4.174/images/{}" /></td></tr>
                <tr><td><img src="http://104.237.4.174/images/{}" /></td><td><img src="http://104.237.4.35/images/{}" /></td></tr>
                <tr><td><img src="http://104.237.4.171/images/{}" /></td><td><img src="http://104.237.4.171/images/{}" /></td></tr>
                <tr><td colspan=2 align="center"><img src="http://104.237.4.91/images/{}" /></td></tr>
                <tr><td colspan=2 align="center"><img src="http://104.237.4.174/images/{}" /></td></tr>
                </table>`);
        }});
        </script>
        """.format(pair, key, key, pair, forecast_tag, 11 + count, "".join(rows), mov_table, pair, key, key, key, img15, img16, img18, img19, img21, img20, img22, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img12, img13, img14, img11))
        count += 1

    for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:

        img19 = "{}_related_pairs.png".format(select_currency)
        img20 = "{}_news_correlation.png".format(select_currency)


        if time_frame == 30:
            forecast_tag = " Monthly Forecast"
        elif time_frame == 7:
            forecast_tag = " Weekly Forecast"
        else:
            forecast_tag = " Daily Forecast"


        key = select_currency + str(time_frame)
        pair_rows.append("""
         <div class="panel panel-default" id="{}">
            <div class="panel-heading" role="tab" id="heading_{}">
                <a id="link_{}" class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion_3" aria-expanded="false">
                    <div class="icon-ac-wrap pr-20"><span class="plus-ac"><i class="fa fa-plus"></i></span><span class="minus-ac"><i class="fa fa-minus"></i></span></div>
                    <B>{}</B> {} 
                </a>
            </div>
            <div id="collapse_{}" class="panel-collapse collapse" role="tabpanel">
                <div class="panel-body pa-15"> 

                <div id="content_{}"></div> 
                </div>
            </div>
        </div>

        <script>
        $('#link_{}').click(function(){{
            $('#content_{}').html(`
                <table border=0 style="width:100%;">
                <tr><td colspan=2><center><img src="http://104.237.4.171/images/{}" /></center></td></tr>
                <tr><td colspan=2><center><img src="http://104.237.4.171/images/{}" /></center></td></tr>
                </table>`);
        }});
        </script>
        """.format(select_currency, key, key, select_currency, forecast_tag, 11 + count, key, key, key, img19, img20))
        count += 1


    return pair_rows, probabilities, pairs, count

def generate_table(signals, rows, pairs, cap):
    real_table = ''.join(rows)

    new_demo_table = ''
    new_real_table = ''
    
    for is_demo in [False]:
        offset = 0

        new_table = ''
        for index in range(len(real_table)):
            if real_table[index] == '?':

                if (offset % 10) != 0 and is_demo:
                    new_table += "<td>" + real_table[index] + "</td>"
                else:

                    if abs(signals[offset]) > cap:
                        if signals[offset] > 0:
                            new_table += "<td style='color:green'>" + str(signals[offset]) + "</td>"
                        else:
                            new_table += "<td style='color:red'>" + str(signals[offset]) + "</td>"
                    else:
                        new_table += "<td></td>"

                offset += 1
            else:
                new_table += real_table[index]

        if is_demo:
            new_demo_table += new_table
        else:
            new_real_table += new_table
    
    return new_real_table

def save_signals_json():

    price_trends_diff = pickle.load(open("{}price_deltas.pickle".format(root_dir), "rb"))

    signals_map = {}
    count = 0
    for time_frame in [30, 7, 1]:

        rows, probabilities, pairs, count = generate_rows(time_frame, price_trends_diff, count)
        new_real_table2 = generate_table(probabilities, rows, pairs, 0)

        signals_map["actual_signals_" + str(time_frame)] = new_real_table2

    signals_map["global_ranking"] = price_trends_diff["global_ranking"]
    signals_map["summary_table"] = price_trends_diff["summary_table"]
    signals_map["indicator_summary"] = price_trends_diff["indicator_summary"]
    signals_map["straddle_summary"] = price_trends_diff["straddle_summary"]

    with open("/var/www/html/signals.json", 'w') as outfile:
        json.dump(signals_map, outfile)


trade_logger = setup_logger('first_logger', root_dir + "news_signal_summary_{}.log".format(sys.argv[1]))

trade_logger.info('Starting ') 

'''
M9: 104.237.4.171
M18: 104.237.4.174
'''

try:
    
    '''
    time_pip_diff = {}
    if sys.argv[1] == "barrier":
        for pair in currency_pairs:
            time_pip_diff[pair] = store_live_history(pair, cnx, "/root/news_signal_all_", "M1", True)
    elif sys.argv[1] == "news_momentum":
        for pair in currency_pairs:
            time_pip_diff[pair] = store_live_history(pair, cnx, "/root/news_momentum_all_", "M3", False)
    elif sys.argv[1] == "time_regression":
        for pair in currency_pairs:
            time_pip_diff[pair] = store_live_history(pair, cnx, "/root/time_regression_all_", "M6", False)
    elif sys.argv[1] == "news_impact":
        for pair in currency_pairs:
            time_pip_diff[pair] = store_live_history(pair, cnx, "/root/news_impact_all_", "M9", True)
    elif sys.argv[1] == "news_reaction_regression":
        for pair in currency_pairs:
            time_pip_diff[pair] = store_live_history(pair, cnx, "/root/news_reaction_regression_all_", "M12", False)
    elif sys.argv[1] == "news_reaction_barrier":
        for pair in currency_pairs:
            time_pip_diff[pair] = store_live_history(pair, cnx, "/root/news_reaction_barrier_all_", "M15", True)
    elif sys.argv[1] == "ranking_regression":
        for pair in currency_pairs:
            time_pip_diff[pair] = store_live_history(pair, cnx, "/root/ranking_regression_all_", "M18", False)
    elif sys.argv[1] == "ranking_barrier":
        for pair in currency_pairs:
            time_pip_diff[pair] = store_live_history(pair, cnx, "/root/ranking_barrier_all_", "M21", True)
    else:
        sys.exit(0)
    '''
    
    if sys.argv[1] == "barrier":
        print ("here")
        save_signals_json()
except:
    trade_logger.info(traceback.format_exc())


trade_logger.info('Finished ') 




