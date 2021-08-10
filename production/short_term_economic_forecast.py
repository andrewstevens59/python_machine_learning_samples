import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle
import glob
from datetime import timedelta
from lxml.html import fromstring
from itertools import cycle
import datetime as dt
import traceback
from io import BytesIO
from scipy import stats
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
import json


import os
import bisect

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
import seaborn as sns
import psutil
import logging
import datetime as dt
from uuid import getnode as get_mac
import traceback

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

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

def get_time_series(symbol, time, granularity="H1"):

    response_buffer = BytesIO() 
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


    for index in range(len(j)):
        item = j[index]

        s = item['time']
        s = s[0 : s.index('.')]
        timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

        if is_valid_trading_period(timestamp):
            times.append(timestamp)
            prices.append(item['closeMid'])
  

    return prices, times

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

checkIfProcessRunning('train_price_action_model.py', "")

if get_mac() != 150538578859218:
    root_dir = "/root/" 
else:
    root_dir = "" 


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

def create_news_table():
    json_map = {}
    for pair in currency_pairs:
        forecasts = pickle.load(open("{}simple_short_forecast_all_{}.pickle".format(root_dir, pair), "rb"))[pair]
        
        grouped_forecasts = {}
        for forecast in forecasts:

            if math.isnan(forecast["forecast"]):
                continue

            key = forecast["description"] + forecast["currency"]
            if key not in grouped_forecasts:
                grouped_forecasts[key] = []

            grouped_forecasts[key].append(forecast)

        table_str = "<table class='table-bordered table-striped' style='border: 1px solid black;width:100%;'>"
        table_str += """
            <tr><th>Pair</th><th>Currency</th><th>Description</th>
            <th>6 Hrs</th>
            <th>12 Hrs</th>
            <th>18 Hrs</th>
            <th>24 Hrs</th>
            <th>30 Hrs</th>
            <th>36 Hrs</th>
            <th>42 Hrs</th>
            <th>48 Hrs</th>
            </tr>
            """
        

        percentiles_by_time = {}
        for percenitle_index in range(8):
            percentiles_by_time[percenitle_index] = []

        for key in grouped_forecasts:

            forecasts = sorted(grouped_forecasts[key], key=lambda x: x["time_wait"])
            forecast = forecasts[0]

            percentiles = []
            for percenitle_index in range(8):
                percentiles.append(int(100 * forecasts[percenitle_index]["global_percentile"]) if forecasts[percenitle_index]["forecast"] > 0 else -int(100 * forecasts[percenitle_index]["global_percentile"]))

                if math.isnan(percentiles[-1]) == False:
                    percentiles_by_time[percenitle_index].append(percentiles[-1])

            table_str += """<tr>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                </tr>
                """.format(pair, 
                    forecast["currency"], 
                    forecast["description"], 
                    ("<font color='lime'>" if percentiles[0] > 0 else "<font color='red'>") + str(percentiles[0]) + "%</font>",
                    ("<font color='lime'>" if percentiles[1] > 0 else "<font color='red'>") + str(percentiles[1]) + "%</font>",
                    ("<font color='lime'>" if percentiles[2] > 0 else "<font color='red'>") + str(percentiles[2]) + "%</font>",
                    ("<font color='lime'>" if percentiles[3] > 0 else "<font color='red'>") + str(percentiles[3]) + "%</font>",
                    ("<font color='lime'>" if percentiles[4] > 0 else "<font color='red'>") + str(percentiles[4]) + "%</font>",
                    ("<font color='lime'>" if percentiles[5] > 0 else "<font color='red'>") + str(percentiles[5]) + "%</font>",
                    ("<font color='lime'>" if percentiles[6] > 0 else "<font color='red'>") + str(percentiles[6]) + "%</font>",
                    ("<font color='lime'>" if percentiles[7] > 0 else "<font color='red'>") + str(percentiles[7]) + "%</font>")

        for percenitle_index in range(8):
            val = np.mean(percentiles_by_time[percenitle_index])
            if math.isnan(val) == False:
                percentiles_by_time[percenitle_index] = int(val)
            else:
                percentiles_by_time[percenitle_index] = 0

        table_str += """<tr>
                 <td>{}</td>
                <td colspan=2>Average</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                </tr>
                """.format(
                    pair, 
                    ("<font color='lime'>" if percentiles_by_time[0] > 0 else "<font color='red'>") + str(percentiles_by_time[0]) + "%</font>",
                    ("<font color='lime'>" if percentiles_by_time[1] > 0 else "<font color='red'>") + str(percentiles_by_time[1]) + "%</font>",
                    ("<font color='lime'>" if percentiles_by_time[2] > 0 else "<font color='red'>") + str(percentiles_by_time[2]) + "%</font>",
                    ("<font color='lime'>" if percentiles_by_time[3] > 0 else "<font color='red'>") + str(percentiles_by_time[3]) + "%</font>",
                    ("<font color='lime'>" if percentiles_by_time[4] > 0 else "<font color='red'>") + str(percentiles_by_time[4]) + "%</font>",
                    ("<font color='lime'>" if percentiles_by_time[5] > 0 else "<font color='red'>") + str(percentiles_by_time[5]) + "%</font>",
                    ("<font color='lime'>" if percentiles_by_time[6] > 0 else "<font color='red'>") + str(percentiles_by_time[6]) + "%</font>",
                    ("<font color='lime'>" if percentiles_by_time[7] > 0 else "<font color='red'>") + str(percentiles_by_time[7]) + "%</font>")

        table_str += "</table>"
        json_map[pair] = table_str

        with open("/var/www/html/short_term_news_summary_{}.html".format(pair), "w") as text_file:
            text_file.write(table_str)

    with open("/var/www/html/short_term_news_summary.json", 'w') as outfile:
        json.dump(json_map, outfile)

def get_calendar_df(pair, year): 

    if pair != None:
        currencies = [pair[0:3], pair[4:7]]
    else:
        currencies = None

    if get_mac() == 150538578859218:
        with open("/Users/andrewstevens/Downloads/economic_calendar/calendar_" + str(year) + ".txt") as f:
            content = f.readlines()
    else:
        with open("/root/trading_data/calendar_" + str(year) + ".txt") as f:
            content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    lines = [x.strip() for x in content] 

    from_zone = tz.gettz('US/Eastern')
    to_zone = tz.gettz('UTC')

    contents = []

    for line in lines:
        line = line[len("2018-12-23 22:44:55 "):]
        toks = line.split(",")

        if currencies == None or toks[1] in currencies:

            time = int(toks[0])
            actual = unicode(toks[3], 'utf-8')
            forecast = unicode(toks[4], 'utf-8')
            previous = unicode(toks[5], 'utf-8')

            actual = "".join([v for v in actual if v.isnumeric() or v in ['.', '+', '-']])
            if len(actual) == 0:
                continue

            try:
                actual = float(actual)

                forecast = "".join([v for v in forecast if v.isnumeric() or v in ['.', '+', '-']])
                if len(forecast) > 0:
                    forecast = float(forecast)
                else:
                    forecast = actual

                previous = "".join([v for v in previous if v.isnumeric() or v in ['.', '+', '-']])
                if len(previous) > 0:
                    previous = float(previous)
                else:
                    previous = actual

                contents.append([toks[1], time, toks[2], actual, forecast, previous, int(toks[6]), toks[7]])
            except:
                pass

    return pd.DataFrame(contents, columns=["currency", "time", "description", "actual", "forecast", "previous", "impact", "better_worse"])

def calculate_time_diff(now_time, ts):

    date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    date = datetime.datetime.utcfromtimestamp(now_time).strftime('%Y-%m-%d %H:%M:%S')
    e = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    _diff = (e - s)

    while s < e:
        max_hour = 24
        if s.day == e.day:
            max_hour = e.hour

        if s.weekday() in {4}:
            max_hour = 21

        if s.weekday() in {4} and s.hour in {21, 22, 23}:
            hours = 1
            _diff -= timedelta(hours=hours)
        elif s.weekday() in {5}:
            hours = max_hour - s.hour
            _diff -= timedelta(hours=hours)
        elif s.weekday() in {6} and s.hour < 21:
            hours = min(21, max_hour) - s.hour
            _diff -= timedelta(hours=hours)
        else:
            hours = max_hour - s.hour

        if hours == 0:
            break
        s += timedelta(hours=hours)

    return (_diff.total_seconds() / (60 * 60))


pattern = re.compile("(1[012]|[1-9]):[0-5][0-9](\\s)?(?i)(am|pm)")

def get_calendar_data(date):

    url='https://www.forexfactory.com/calendar.php?day=' + date
    #Create a handle, page, to handle the contents of the website
    from selenium import webdriver
    driver = webdriver.Firefox(executable_path = '/usr/local/bin/geckodriver')

    print (url)
    driver.get(url)
    page_source = driver.page_source
    driver.close()

    time_zone_inc = page_source.find("Calendar Time Zone")
    time_zone_inc = page_source[time_zone_inc:time_zone_inc+200].find("GMT ") + time_zone_inc
    
    time_zone_inc += len("GMT ")
    time_component = page_source[time_zone_inc:time_zone_inc+200]
    time_offset = int(time_component.split(')')[0]) * 60 * 60
    time_offset += (60 * 60)

    #Store the contents of the website under doc
    doc = lh.fromstring(page_source)
    #Parse data that are stored between <tr>..</tr> of HTML
    tr_elements = doc.xpath('//tr')



    contents = []
    currencies = ["GBP", "USD", "AUD", "CAD", "NZD", "JPY", "CHF", "EUR"]

    curr_time = None
    #Since out first row is the header, data is stored on the second row onwards
    for j in range(0,len(tr_elements)):
        #T is our j'th row
        T=tr_elements[j]




        found_currency = False
        found_description = False

        better_worse = "?"
        impact = 0
        actual = None
        forecast = None
        previous = None
        space = None
        space1 = None
        currency = None
        description = None
        #Iterate through each element of the row
        for t in T.iterchildren():

            class_name = t.get('class')
            if class_name != None:
                if "calendar__impact--low" in class_name:
                    impact = 1
                if "calendar__impact--medium" in class_name:
                    impact = 2
                if "calendar__impact--high" in class_name:
                    impact = 3

            html = lh.tostring(t)
            if html != None:
                if 'class="better"' in str(html):
                    better_worse = "B"
                if 'class="worse"' in str(html):
                    better_worse = "W"
      
            data=t.text_content().strip()

            if found_currency == True and space1 == None:
                space1 = data
                continue

            if found_currency == True:
                found_currency = False
                found_description = True
                description = data

                continue

            if found_description == True:

                if space == None:
                    space = data
                    print (data, "Space")
                    continue

                if actual == None:
                    actual = data
                    print (data, "Actual")
                    continue

                if forecast == None:
                    forecast = data
                    print (data, "Forecast")
                    continue

                if previous == None:
                    previous = data
                    print (previous, "Previous")
                    print (description, "description")

                    utc = datetime.datetime.strptime(date + " " + curr_time, "%b%d.%Y %I:%M%p")
                    timestamp = calendar.timegm(utc.timetuple())
                    timestamp -= time_offset

                    actual = str(actual)
                    forecast = str(forecast)
                    previous = str(previous)

                    print (actual, forecast, previous)

                    actual = "".join([v for v in actual if v.isnumeric() or v in ['.', '+', '-']])
                    if len(actual) == 0:
                        continue

                    try:
                        actual = float(actual)

                        forecast = "".join([v for v in forecast if v.isnumeric() or v in ['.', '+', '-']])
                        if len(forecast) > 0:
                            forecast = float(forecast)
                        else:
                            forecast = actual

                        previous = "".join([v for v in previous if v.isnumeric() or v in ['.', '+', '-']])
                        if len(previous) > 0:
                            previous = float(previous)
                        else:
                            previous = actual

                        contents.append([currency, timestamp, description, actual, forecast, previous, impact, better_worse])
                    except:
                        pass

                    
                    #print (str(timestamp) + "," + currency + "," + description + "," + actual + "," + forecast + "," + previous + "," + str(impact)) 
                    #trade_logger.info(str(timestamp) + "," + currency + "," + description + "," + actual + "," + forecast + "," + previous + "," + str(impact)) 
                    continue

            if pattern.match(data):
                curr_time = data

            if data in currencies:
                print (date, curr_time, data)
                found_currency = True
                currency = data

    return pd.DataFrame(contents, columns=["currency", "time", "description", "actual", "forecast", "previous", "impact", "better_worse"])

news_release_stat_df = pd.read_csv("../news_dist_stats.csv".format(root_dir))
news_release_stat_df.set_index("key", inplace=True)

def create_correlation_graph(calendar_df, pdf):

    for i, compare_pair in enumerate(currency_pairs):
        prices, times = get_time_series(compare_pair, 24 * 20 * 2)
        before_price_df2 = pd.DataFrame()
        before_price_df2["prices" + str(i)] = prices
        before_price_df2["times"] = times

        if i > 0:
            before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
            before_all_price_df.reset_index(inplace=True)
        else:
            before_all_price_df = before_price_df2


    times = before_all_price_df["times"].values.tolist()
    

    for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:

        print (select_currency)
        correlation_times_frame = {}
        timestamps_times_frame = {}
        

        for hours_back in range(1, 24 * 20):

            for time_frame in [4, 8, 16, 32]:
                

                correlations = []
                for i, compare_pair1 in enumerate(currency_pairs):
                    if select_currency not in compare_pair1:
                        continue

                    pair1 = before_all_price_df["prices" + str(i)].values.tolist()

                    if compare_pair1[0:3] != select_currency:
                        pair1 = [1.0 / price for price in pair1]

                    for j, compare_pair2 in enumerate(currency_pairs):
                        if j <= i:
                            continue

                        if select_currency not in compare_pair2:
                            continue

                        pair2 = before_all_price_df["prices" + str(j)].values.tolist()
                        if compare_pair2[0:3] != select_currency:
                            pair2 = [1.0 / price for price in pair2]
                    
                        correlation, p_value = stats.pearsonr(pair1[(-time_frame*24) - hours_back:-hours_back], pair2[(-time_frame*24) - hours_back:-hours_back])

                        correlations.append(correlation)

                if time_frame not in correlation_times_frame:
                    correlation_times_frame[time_frame] = []
                    timestamps_times_frame[time_frame] = []

                correlation_times_frame[time_frame].append(np.mean(correlations))
                timestamps_times_frame[time_frame].append(times[-hours_back])

        time_subset = times[-hours_back:]
        sub_df = calendar_df[calendar_df["currency"] == select_currency]
        timestamps = []
        z_scores = []
        impacts = []
        impact_set = set()

        for index, row in sub_df.iterrows():
            key = row["description"] + "_" + row["currency"]
            stat_row = news_release_stat_df[news_release_stat_df.index == key]
            if len(stat_row) > 0:
                stat_row = stat_row.iloc[0]

                sign = stat_row["sign"]

                if stat_row["forecast_std"] > 0:
       
                    z_score1 = ((float(row["actual"]) - float(row["forecast"])) - stat_row["forecast_mean"]) / stat_row["forecast_std"]
                else:
                    z_score1 = 1

                z_score1 = min(z_score1, 2.5)

            else:
                z_score1 = 1

    
            if z_score1 is not None and row["time"] >= time_subset[0]:
                timestamps.append(row["time"])
                z_scores.append(z_score1)
                impacts.append(row["impact"])
                impact_set.add(row["impact"])


        plt.title(select_currency + " Basket Correlation Plot")

        for time_frame in [4, 8, 16, 32]:
            plt.plot([bisect.bisect(time_subset, timestamp) for timestamp in timestamps_times_frame[time_frame]], correlation_times_frame[time_frame], label="Time Frame {} Days".format(time_frame))
            
        for impact in impact_set:
            select_z_scores = [abs(z) / 10 for z, i in zip(z_scores, impacts) if i == impact]
            select_timestamps = [t for t, i in zip(timestamps, impacts) if i == impact]

            if impact == 1:
                impact_label = "Low"
                color_code = "darkseagreen"
            elif impact == 2:
                impact_label = "Medium"
                color_code = "palegreen"
            elif impact == 3:
                impact_label = "High"
                color_code = "lime"
            else:
                impact_label = "Undefined"
                color_code = "yellow"


 
            plt.plot([bisect.bisect(time_subset, timestamp) for timestamp in select_timestamps], select_z_scores, 'o', color=color_code, label="Economic News Impact {}".format(impact_label))


        selectes_times = []
        x_tick_indexes = []
        prev_time = None
        date_range = [[datetime.datetime.utcfromtimestamp(time_subset[t]).strftime('%m-%d'), bisect.bisect(time_subset, time_subset[t])] for t in range(len(time_subset))]
        for item in date_range:

            if len(x_tick_indexes) > 0 and abs(item[1] - x_tick_indexes[-1]) < 24:
                continue

            if item[0] != prev_time:
                selectes_times.append(item[0])
                x_tick_indexes.append(item[1])
                prev_time = item[0]


        plt.ylabel("Correlation")
        plt.xticks(x_tick_indexes, selectes_times, rotation=30)

        plt.legend()
        #plt.savefig("/var/www/html/images/{}_news_correlation.png".format(select_currency))
        pdf.savefig()
        #plt.show()
        plt.close()

        for i, compare_pair in enumerate(currency_pairs):
            if select_currency not in compare_pair:
                continue

            prices = []
            for hours_back1 in reversed([v for v in range(1, 24 * 20)]):
                pair1 = before_all_price_df["prices" + str(i)].values.tolist()[-hours_back1:]
                display_pair = compare_pair
                if compare_pair[0:3] != select_currency:
                    pair1 = [1.0 / price for price in pair1]
                    display_pair = compare_pair[4:7] + "_" + compare_pair[0:3]

                mean = np.mean(pair1)
                std = np.std(pair1)

                prices.append((pair1[-1] - mean) / std)

            plt.plot([bisect.bisect(time_subset, timestamp) for timestamp in time_subset], prices, label="Pair {}".format(display_pair))

        plt.title("Basket Position {}".format(select_currency))
        plt.xticks(x_tick_indexes, selectes_times, rotation=30)
        plt.legend()
        #plt.savefig("/var/www/html/images/{}_related_pairs.png".format(select_currency))
        pdf.savefig()
        #plt.show()
        plt.close()

        
        for i, compare_pair in enumerate(currency_pairs):
            if select_currency not in compare_pair:
                continue

            pair1 = before_all_price_df["prices" + str(i)].values.tolist()[-hours_back:]
            display_pair = compare_pair
            if compare_pair[0:3] != select_currency:
                pair1 = [1.0 / price for price in pair1]
                display_pair = compare_pair[4:7] + "_" + compare_pair[0:3]

            mean = np.mean(pair1)
            std = np.std(pair1)

            pair1 = [(p - mean) / std for p in pair1]

            plt.plot([bisect.bisect(time_subset, timestamp) for timestamp in time_subset], pair1, label="Pair {}".format(display_pair))

        plt.title("Related Pairs {}".format(select_currency))
        plt.xticks(x_tick_indexes, selectes_times, rotation=30)
        plt.legend()
        #plt.savefig("/var/www/html/images/{}_related_pairs.png".format(select_currency))
        pdf.savefig()
        #plt.show()
        plt.close() 




dates = sorted(glob.glob("calendar/*"))
dates = [[d, datetime.datetime.strptime(d[len("calendar/"):], "%b%d.%Y")] for d in dates]
dates = sorted(dates, key=lambda x: x[1], reverse=True)
dates = [v[0][len("calendar/"):] for v in dates][:3]
dates = []

dfs = []
for back_day in range(-1, 30):
    d = datetime.datetime.now() - datetime.timedelta(days=back_day)

    day_before = d.strftime("%b%d.%Y").lower()

    if os.path.isfile("calendar/" + day_before) == False or str(day_before) in dates:
        df = get_calendar_data(day_before) 
        df.to_csv("calendar/" + day_before, index=False)
    else:
        df = pd.read_csv("calendar/" + day_before)

    dfs.append(df)

with PdfPages('multipage_pdf.pdf') as pdf:
    create_correlation_graph(pd.concat(dfs), pdf)




