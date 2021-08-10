import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle
from dateutil import tz
import calendar
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
import json


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]



def checkIfProcessRunning(processName, command):
    count = 0
    #Iterate over the all the running process
    for proc in psutil.process_iter():

        try:
            cmdline = proc.cmdline()

            # Check if process name contains the given name string.
            if len(cmdline) > 1 and processName.lower() in cmdline[1]: 
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if count >= 2:
        sys.exit(0)

checkIfProcessRunning('execute_news_signals_plots.py', "")


cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

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

if os.path.isfile("trend_bias.pickle"):
    trend_bias_map = pickle.load(open("trend_bias.pickle", "rb"))
else:
    trend_bias_map = {}

if os.path.isfile("barrier_bias.pickle"):
    barrier_bias_map = pickle.load(open("trend_bias.pickle", "rb"))
else:
    barrier_bias_map = {}

def get_forecast_bias(pair, time_frame):

    return 0

    key = pair + "_" + str(int(time_frame / 24))

    if key in trend_bias_map:
        return trend_bias_map[key]

    print ("forecast", time_frame)

    prices, times, volumes = load_time_series(pair, None, False)

    trends = []
    for index1 in range(0, len(prices) - time_frame, 24):
        trends.append(prices[time_frame + index1] - prices[index1])

    bias = np.mean(trends)
    trend_bias_map[key] = bias

    print ("bias", bias)

    return bias

def get_barrier_bias(pair, barrier):

    return 0.5

    key = pair + "_" + str(barrier)

    if key in barrier_bias_map:
        return barrier_bias_map[key]

    if pair[4:7] == "JPY":
        pip_size = 0.01
    else:
        pip_size = 0.0001

    print ("barrier", barrier)

    prices, times, volumes = load_time_series(pair, None, False)

    trends = []
    for index in range(0, len(prices), 24):

        after_prices = prices[index:]

        top_barrier = after_prices[0] + (barrier * pip_size)
        bottom_barrier = after_prices[0] - (barrier * pip_size)

        found = None
        for price in after_prices:
            if price >= top_barrier:
                found = 1
                break

            if price <= bottom_barrier:
                found = 0
                break

        if found != None:
            trends.append(found)

    bias = np.mean(trends)
    barrier_bias_map[key] = bias
    print ("bias", bias)

    return bias

def load_model_distributions_binary_error(cnx, pair, model_types):

    dist_map = {}
    dist_map["auc"] = {"All" : []}
    dist_map["rmse"] = {"All" : []}
    dist_map["binary_error"] = {"All" : []}

    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

    query = ("""
        SELECT auc, rmse, model_type, TF, FT, TF + FT + TT + FF
        FROM model_historical_stats 
        where pair = '{}'
        and model_type in ({})
        and timestamp >= {} - (60 * 60 * 24 * 60)
        """.format(pair, model_types, time.time()))

    cursor = cnx.cursor()
    cursor.execute(query)

    for row1 in cursor:
        auc = row1[0]
        rmse = row1[1]
        if rmse < 0 or auc < 0:
            continue

        model_type = row1[2]
        TF = float(row1[3])
        FT = float(row1[4])
        total_binary = float(row1[5])
        if total_binary == 0:
            continue

        TF = (float(TF) / total_binary) * 100
        FT = (float(FT) / total_binary) * 100

        dist_map["binary_error"]["All"].append(TF)
        dist_map["binary_error"]["All"].append(FT)

        if auc != 0:
            dist_map["auc"]["All"].append(auc)

        if rmse != 0:
            dist_map["rmse"]["All"].append(rmse)

    dist_map["binary_error"]["All"] = sorted(dist_map["binary_error"]["All"])
    dist_map["auc"]["All"] = sorted(dist_map["auc"]["All"])
    dist_map["rmse"]["All"] = sorted(dist_map["rmse"]["All"])
    return dist_map

def load_model_distributions(cnx, pair, model_types):

    dist_map = {}
    dist_map["auc"] = {"All" : []}
    dist_map["rmse"] = {"All" : []}

    dist_map["prob"] = {"All" : []}
    dist_map["forecast"] = {"All" : []}

    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')


    query = ("""
        SELECT auc, rmse, barrier, model_type, prob, forecast
        FROM model_historical_stats 
        where pair = '{}'
        and model_type in ({})
        and timestamp >= {} - (60 * 60 * 24 * 60)
        """.format(pair, model_types, time.time()))

    cursor = cnx.cursor()
    cursor.execute(query)

    for row1 in cursor:
        auc = row1[0]
        rmse = row1[1]
        if rmse < 0 or auc < 0:
            continue

        barrier = int(row1[2])
        model_type = row1[3]
        prob = row1[4]
        forecast = row1[5]

        prob = prob - get_barrier_bias(pair, barrier)
        forecast = forecast - get_forecast_bias(pair, barrier)

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


def show_model_plots(pdf, pair, cnx, query, title, dist_map, is_barrier_model, is_all, time_frame):


    cursor = cnx.cursor()
    cursor.execute(query)

    prob_zero_perc = bisect(dist_map["prob"]["All"], 0)
    forecast_zero_perc = bisect(dist_map["forecast"]["All"], 0)

    #prob - 0.5, auc, barrier, rmse, forecast, model_type, description, currency

    forecast_count = 0
    norm_count = 0

    model_probs = {}
    for row1 in cursor:

        prob = row1[0]
        auc = row1[1]
        barrier = int(row1[2])
        rmse = row1[3]
        forecast = row1[4]
        model_type = row1[5]
        description = row1[6] + "_" + row1[7] + "_" + str(row1[8])

        prob = prob - get_barrier_bias(pair, barrier)
        forecast = forecast - get_forecast_bias(pair, barrier)

        if model_type not in model_probs:
            model_probs[model_type] = {}

        if rmse != 0:

            forecast_percentile = float(bisect(dist_map["forecast"]["All"], forecast) - forecast_zero_perc) / len(dist_map["forecast"]["All"])

            if rmse > 0:
                norm_count += 1
                if description + "_rmse_norm" not in model_probs[model_type]:
                    model_probs[model_type][description + "_rmse_norm"] = []

                rmse_percentile = (float(bisect(dist_map["rmse"]["All"], rmse)) / len(dist_map["rmse"]["All"]))
                model_probs[model_type][description + "_rmse_norm"].append(forecast_percentile / (1 + rmse_percentile))
            else:
                forecast_count += 1
                if description + "_rmse_forecast" not in model_probs[model_type]:
                    model_probs[model_type][description + "_rmse_forecast"] = []

                model_probs[model_type][description + "_rmse_forecast"].append(forecast_percentile)

        if is_barrier_model:

            prob_percentile = (float(bisect(dist_map["prob"]["All"], prob) - prob_zero_perc) / len(dist_map["prob"]["All"])) 

            if auc > 0:
                norm_count += 1
                if description + "_auc_norm" not in model_probs[model_type]:
                    model_probs[model_type][description + "_auc_norm"] = []

                auc_percentile = 1.0 - (min(0.2, max(0, auc - 0.5)) / 0.2)
                model_probs[model_type][description + "_auc_norm"].append(prob_percentile / (1 + auc_percentile))
            else:
                forecast_count += 1
                if description + "_auc_forecast" not in model_probs[model_type]:
                    model_probs[model_type][description + "_auc_forecast"] = []

                model_probs[model_type][description + "_auc_forecast"].append(prob_percentile)

    print ("Stat Count", forecast_count, norm_count)
    sns.set_style("white")

    # Plot
    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

    plt.figure(figsize=(5,4))
    colors = ["dodgerblue", "orange", "deeppink", "lime"]

    percentiles = []
    for model, color in zip(model_probs.keys(), colors):

        probs_10 = [np.percentile(model_probs[model][description], 10) for description in model_probs[model]]
        probs_25 = [np.percentile(model_probs[model][description], 25) for description in model_probs[model]]
        probs_50 = [np.percentile(model_probs[model][description], 50) for description in model_probs[model]]
        probs_75 = [np.percentile(model_probs[model][description], 75) for description in model_probs[model]]
        probs_90 = [np.percentile(model_probs[model][description], 90) for description in model_probs[model]]

        probs = [p for p in probs_10 + probs_25 + probs_50 + probs_75 + probs_90 if abs(p) > 0.1]

        for percentile in range(1, 200):
            percentiles.append(np.percentile(probs, percentile * 0.5))

        if is_all == False:
            if len(percentiles) > 3:
                sns.distplot(percentiles, color=color, label=model, **kwargs)

            percentiles = []

    is_buy = 0
    if is_all == True:
        if len(percentiles) > 3:
            if (np.percentile(percentiles, 60) > 0) == (np.percentile(percentiles, 40) > 0):
                if np.median(percentiles) > 0:
                    is_buy = 1
                    sns.distplot(percentiles, color="lime", label="Economic Forecast BUY", **kwargs)
                else:
                    is_buy = -1
                    sns.distplot(percentiles, color="r", label="Economic Forecast SELL", **kwargs)
            else:
                sns.distplot(percentiles, color="orange", label="Economic Forecast NEURTRAL", **kwargs)
    
        query = ("""INSERT INTO signal_summary(timestamp, pair, model_type, model_group, forecast_dir, forecast_percentiles, time_frame) 
                    values (now(),'{}','{}','{}','{}','{}','{}')""".
            format(
                pair,
                "Barrier" if is_barrier_model else "Forecast",
                "Economic",
                is_buy,
                json.dumps(percentiles),
                time_frame
                ))

        cursor.execute(query)
        cnx.commit()

    plt.xlim(-1,1)
    plt.ylim(0,25)

    if time_frame == 1:
        plt.title(title + " Daily")
    elif time_frame == 7:
        plt.title(title + " Weekly")
    elif time_frame == 30:
        plt.title(title + " Monthly")

    plt.xlabel("SELL <-----> BUY")

    plt.axvline(0, color='black')

    plt.legend()
    plt.savefig("/var/www/html/images/{}_barrier_{}_all_{}_time_frame_{}.png".format(pair, is_barrier_model, is_all, time_frame))
    #pdf.savefig()
    plt.close()




def show_error_plots_dist(pdf, pair, cnx, barrier_models, regression_models, query, title, time_frame):

    dist_map_barrier = load_model_distributions_binary_error(cnx, pair, barrier_models)
    dist_map_regression = load_model_distributions_binary_error(cnx, pair, regression_models)

    cursor = cnx.cursor()
    cursor.execute(query)

    model_probs = {}
    for row1 in cursor:

        TF = float(row1[0])
        FT = float(row1[1])
        total = float(row1[2])
        if total == 0:
            continue

        model_type = row1[3]
        rmse = row1[4]
        auc = row1[5]
        description = row1[6] + "_" + row1[7] + "_" + str(row1[8])
        TF = (float(TF) / total) * 100
        FT = (float(FT) / total) * 100

        if model_type in barrier_models:
            model_type = "Price Levels"
            is_barrier_model = True
            dist_map = dist_map_barrier
        else:
            model_type = "Forecasts"
            is_barrier_model = False
            dist_map = dist_map_regression

        if model_type not in model_probs:
            model_probs[model_type] = {}

        if rmse != 0:
            if description + "_rmse" not in model_probs[model_type]:
                model_probs[model_type][description + "_rmse"] = []

            rmse_percentile = (float(bisect(dist_map["rmse"]["All"], rmse)) / len(dist_map["rmse"]["All"]))

            error_percentile = float(bisect(dist_map["binary_error"]["All"], TF)) / len(dist_map["binary_error"]["All"])
            model_probs[model_type][description + "_rmse"].append(error_percentile / (1 + rmse_percentile))
            error_percentile = float(bisect(dist_map["binary_error"]["All"], FT)) / len(dist_map["binary_error"]["All"])
            model_probs[model_type][description + "_rmse"].append(-error_percentile / (1 + rmse_percentile))

        if is_barrier_model:
            if description + "_auc" not in model_probs[model_type]:
                model_probs[model_type][description + "_auc"] = []

            auc_percentile = 1.0 - (min(0.2, max(0, auc - 0.5)) / 0.2)

            error_percentile = float(bisect(dist_map["binary_error"]["All"], TF)) / len(dist_map["binary_error"]["All"])
            model_probs[model_type][description + "_auc"].append(error_percentile / (1 + auc_percentile))

            error_percentile = float(bisect(dist_map["binary_error"]["All"], FT)) / len(dist_map["binary_error"]["All"])
            model_probs[model_type][description + "_auc"].append(-error_percentile / (1 + auc_percentile))

    sns.set_style("white")

    # Plot
    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

    plt.figure(figsize=(5,4))
    colors = ["dodgerblue", "orange"]

    for model, color in zip(model_probs.keys(), colors):

        probs_25 = [np.percentile(model_probs[model][description], 25) for description in model_probs[model]]
        probs_50 = [np.percentile(model_probs[model][description], 50) for description in model_probs[model]]
        probs_75 = [np.percentile(model_probs[model][description], 50) for description in model_probs[model]]

        probs = [p for p in probs_25 + probs_50 + probs_75 if abs(p) > 0.05]

        if len(probs) > 3:
            sns.distplot(probs, color=color, label=model, **kwargs)

    plt.xlim(-1,1)
    plt.ylim(0,25)

    if time_frame == 1:
        plt.title(title + " Daily")
    elif time_frame == 7:
        plt.title(title + " Weekly")
    elif time_frame == 30:
        plt.title(title + " Monthly")

    plt.xlabel("SELL <-----> BUY")

    plt.axvline(0, color='black')

    plt.legend()
    plt.savefig("/var/www/html/images/{}_error_dist_{}.png".format(pair, time_frame))
    #pdf.savefig()
    plt.close()

def show_error_plots_bar(pdf, pair, cnx, barrier_models, regression_models, query, title, time_frame):

    cursor = cnx.cursor()
    cursor.execute(query)

    rows = [row for row in cursor]

    sns.set_style("white")

    # Plot
    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})
    colors = ["orange", "deeppink"]

    plt.figure(figsize=(5,4))

    x_offset = -0.85

    for row1 in rows:

        TF = float(row1[0])
        FT = float(row1[1])
        total = float(row1[2])
        model_type = row1[3]

        TF = (float(TF) / total) * 100
        FT = (float(FT) / total) * 100

        plt.text(x_offset - 0.035, FT -5, str(int(FT)), color='black')
        plt.text(1 + x_offset - 0.035, TF -5, str(int(TF)), color='black')

        if model_type in barrier_models:
            plt.bar([x_offset, 1 + x_offset], [FT, TF], color = 'dodgerblue', width = 0.1, label='Price Level ' + model_type)
        else:
            plt.bar([x_offset, 1 + x_offset], [FT, TF], color = 'orange', width = 0.1, label='Forecast ' + model_type)

        x_offset += 0.1

    plt.xlim(-1,1)
    plt.ylim(0,100)

    if time_frame == 1:
        plt.title(title + " Daily")
    elif time_frame == 7:
        plt.title(title + " Weekly")
    elif time_frame == 30:
        plt.title(title + " Monthly")

    plt.xlabel("SELL <-----> BUY")
    plt.ylabel("Error (%)")

    plt.axvline(0, color='black')

    plt.legend()
    plt.savefig("/var/www/html/images/{}_error_percentage_time_frame_{}.png".format(pair, time_frame))
    #pdf.savefig()
    plt.close()

def create_plots():

    barrier_models = ",".join(["'M1'", "'M9'", "'M15'", "'M21'"])
    regression_models = ",".join(["'M3'", "'M6'", "'M12'", "'M18'"])

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages('multipage_pdf.pdf') as pdf:

        for pair in currency_pairs:

            print (pair)

            cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

            for time_frame in [30, 7, 1]:
                if time_frame > 1:
                    time_threshold = 4
                else:
                    time_threshold = 7

                show_error_plots_bar(pdf, pair, cnx, 
                    barrier_models, regression_models,
                            ("""SELECT sum(TF), sum(FT), sum(TF + FT + TT + FF), model_type  
                                FROM model_historical_stats 
                                where pair = '{}'
                                and ROUND(({} - timestamp) / (60 * 60 * 24 * {}), 0) < {}
                                and (TF + FT + FF + TT) > 0
                                group by model_type
                                """.format(pair, time.time(), time_frame, time_threshold)),
                            'Error (%) ' + pair, time_frame)

            for model_types in [barrier_models, regression_models]:

                cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

                dist_map = load_model_distributions(cnx, pair, model_types)

                for time_frame in [30, 7, 1]:

                    if time_frame > 1:
                        time_threshold = 4
                    else:
                        time_threshold = 7

                    if model_types == barrier_models:
                        show_model_plots(pdf, pair, cnx, 
                            ("""SELECT prob, auc, barrier, rmse, forecast, model_type, description, currency, timestamp
                                FROM model_historical_stats 
                                where pair = '{}' 
                                and model_type in ({})
                                and ROUND(({} - timestamp) / (60 * 60 * 24 * {}), 0) < {}
                                """.format(pair, model_types, time.time(), time_frame, time_threshold)),
                            'Price Levels ' + pair, dist_map, True, True, time_frame)

                        show_model_plots(pdf, pair, cnx, 
                            ("""SELECT prob, auc, barrier, rmse, forecast, model_type, description, currency, timestamp   
                                FROM model_historical_stats 
                                where pair = '{}'
                                and model_type in ({})
                                and ROUND(({} - timestamp) / (60 * 60 * 24 * {}), 0) < {}
                                """.format(pair, model_types, time.time(), time_frame, time_threshold)),
                            'Price Levels ' + pair, dist_map, True, False, time_frame)

                    else:


                        show_model_plots(pdf, pair, cnx, 
                            ("""SELECT prob, auc, barrier, rmse, forecast, model_type, description, currency, timestamp 
                                FROM model_historical_stats 
                                where pair = '{}'
                                and model_type in ({})
                                and ROUND(({} - timestamp) / (60 * 60 * 24 * {}), 0) < {}
                                """.format(pair, model_types, time.time(), time_frame, time_threshold)),
                            'Forecast ' + pair, dist_map, False, True, time_frame)
                

                        show_model_plots(pdf, pair, cnx, 
                            ("""SELECT prob, auc, barrier, rmse, forecast, model_type, description, currency, timestamp  
                                FROM model_historical_stats 
                                where pair = '{}'
                                and model_type in ({})
                                and ROUND(({} - timestamp) / (60 * 60 * 24 * {}), 0) < {}
                                """.format(pair, model_types, time.time(), time_frame, time_threshold)),
                            'Forecast ' + pair, dist_map, False, False, time_frame)

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

trade_logger = setup_logger('first_logger', root_dir + "news_signal_plots.log")
trade_logger.info('Starting ') 

try:
    create_plots()
    trade_logger.info('Finished ') 
except:
    print (traceback.format_exc())
    trade_logger.info(traceback.format_exc())

 

