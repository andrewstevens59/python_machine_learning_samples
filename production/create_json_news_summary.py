import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle
from dateutil import tz
import calendar
import mysql.connector
from pytz import timezone
import pandas as pd
import numpy as np
import time
import json
import psutil
import logging
import datetime as dt

root_dir = "/root/"


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

checkIfProcessRunning('create_json_news_summary.py', "")

def get_curr_calendar_day():

    curr_date = datetime.datetime.now(timezone('US/Eastern')).strftime("%b%d.%Y").lower()

    week_day = datetime.datetime.now(timezone('US/Eastern')).weekday()

    print ("curr day", week_day)

    
    if os.path.isfile("/tmp/calendar_data_historic_short"):
        calendar = pickle.load(open("/tmp/calendar_data_historic_short", 'rb'))

        news_times = calendar["df"]["time"].values.tolist()

        found_recent_news = False
        for news_time in news_times:
            if abs(time.time() - news_time) < 7 * 60 and time.time() > news_time:
                print ("find new news")
                found_recent_news = True


        if abs(time.time() - calendar["last_check"]) < 60 * 60 * 1:
            if len(calendar["df"]) > 0:
                return calendar["df"]


    if week_day == 6 or week_day == 0:
        back_day_num = 4
    else:
        back_day_num = 2

    print ("loading...", back_day_num)

    lag_days = 5

    calendar_data = []
    for back_day in range(-1, back_day_num + 15):
        d = datetime.datetime.now(timezone('US/Eastern')) - datetime.timedelta(days=back_day)

        day_before = d.strftime("%b%d.%Y").lower()
        print (day_before)

        df = pd.read_csv(root_dir + "news_data/{}.csv".format(day_before))
        calendar_data = [df] + calendar_data

        if len(df) > 0:
            min_time = df["time"].min()
            time_lag_compare = calculate_time_diff(time.time(), min_time)
            print ("time lag", time_lag_compare)
            if time_lag_compare >= 24 * lag_days:
                break


    calendar = {"last_check" : time.time(), "day" :  curr_date, "df" : pd.concat(calendar_data)}

    pickle.dump(calendar, open("/tmp/calendar_data_historic_short", 'wb'))

    return calendar["df"]

def create_news_summary():
    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                      host='mysql.newscaptial.com',
                      database='newscapital')


    query = ("""SELECT pair, prob, auc, barrier, rmse, forecast, model_type, description, currency, ROUND(({} - timestamp), 0) / (60 * 60) diff_hours
                        FROM model_historical_stats
                        where ROUND(({} - timestamp), 0) < (60 * 60 * 24 * 7)
                        order by timestamp
                        """.format(time.time(), time.time()))

    cursor = cnx.cursor()
    cursor.execute(query)

    pair_summary = {}

    pair_summary["calendar"] = get_curr_calendar_day().to_dict()

    for row1 in cursor:
        pair = row1[0]
        prob = row1[1]
        auc = row1[2]
        barrier = int(row1[3])
        rmse = row1[4]
        forecast = row1[5]
        model_type = row1[6]
        description = row1[7]
        currency = row1[8]
        diff_hours = int(row1[9])

        if pair not in pair_summary:
            pair_summary[pair] = []

        pair_summary[pair].append({
            "prob" : prob,
            "auc" : auc,
            "barrier" : barrier,
            "rmse" : rmse,
            "forecast" : forecast,
            "model_type" : model_type,
            "description" : description,
            "currency" : currency,
            "diff_hours" : diff_hours
            })

    with open('/var/www/html/news_summary.json', 'w') as fp:
        json.dump(pair_summary, fp)



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

trade_logger = setup_logger('first_logger', root_dir + "create_json_news_summary.log")
trade_logger.info('Starting ')

try:
    create_news_summary()
    trade_logger.info('Finished ')
except:
    print (traceback.format_exc())
    trade_logger.info(traceback.format_exc())