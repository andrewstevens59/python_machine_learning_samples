import ast
import json
import time
import sys
import glob
import calendar
import datetime
from dateutil.tz import *
import matplotlib


matplotlib.use('Agg')

files = glob.glob("/root/trade_news_release*.log")
to_zone = tzutc()
from matplotlib import pyplot as plt

def tail( f, lines=20 ):
    total_lines_wanted = lines

    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = [] # blocks of size BLOCK_SIZE, in reverse order starting
                # from the end of the file
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            # read the last block we haven't yet read
            f.seek(block_number*BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count('\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = ''.join(reversed(blocks))
    return all_read_text.splitlines()[-total_lines_wanted:]


sorted_set = []
for file_name in files:

    if "exponential" in file_name or "low_barrier" in file_name:
        continue

    print file_name

    text_rows = []
    time_start = None
    with open(file_name, "r") as fi:
        text_rows = tail(fi, 100000)

    lines = []
    index = 0
    skip_ahead = False
    while index < len(text_rows):
        line = text_rows[index]
        index += 1

        if len(lines) > 0 and skip_ahead == False:
            index = len(text_rows) - 50
            skip_ahead = True
    
        ln = line[len("2018-09-21 12:23:05 "):]

        if ln.startswith("Equity:"):
            time_start = calendar.timegm(datetime.datetime.strptime(line[:len("2018-09-21 12:23:05")], "%Y-%m-%d %H:%M:%S").timetuple())

            if time.time() - time_start < sys.argv[1] * 60 * 60 * 24:
                lines.append(float(ast.literal_eval(ln[len("Equity: "):])))

        if ln.startswith("Max Value:"):
            local = datetime.datetime.strptime(line[:len("2018-09-21 12:23:05")], "%Y-%m-%d %H:%M:%S")
            local = local.replace(tzinfo=tzlocal())
            utc = local.astimezone(to_zone)
            finish_time = calendar.timegm(utc.timetuple())

    sorted_set.append([file_name, (lines[-1] / lines[0]), lines[0], lines[-1], finish_time])

    if time.time() - finish_time > 60 * 60:
        print (time.time() - finish_time) / (60 * 60), "error"

sorted_set = sorted(sorted_set, key=lambda x: abs(x[1]), reverse=False)

errors = []

image_names = []
image_count = 0
for item in sorted_set:
    print (item[0], (item[1] - 1.0) * 100, item[2], item[3])

    if item[1] > 1:

        equity_curve = []
        text_rows = []
        with open(item[0], "r") as fi:
            text_rows = tail(fi, 100000)

        for line in text_rows:
            ln = line[len("2018-09-21 12:23:05 "):]

            curr_time = calendar.timegm(datetime.datetime.strptime(line[:len("2018-09-19 10:08:08")], "%Y-%m-%d %H:%M:%S").timetuple())

            if ln.startswith("Equity:"):
                equity_curve.append(float(ast.literal_eval(ln[len("Equity: "):])))

        plt.figure(figsize=(6, 4))
        plt.plot(equity_curve)
        plt.title(item[0])

        plt.savefig("/var/www/html/" + str(image_count) + ".png")  # saves the current figure into a pdf page
        plt.close() 
        
        image_names.append("<h3>" + item[0] + "</h3><br><img src='" + str(image_count) + ".png' /><br>")
        image_count += 1

    if time.time() - item[4] > 60 * 60:
	   errors.append(item[0])


from tzlocal import get_localzone # $ pip install tzlocal
from_zone = get_localzone()

for file_name in ["update_news_release_signals_allAUD_NZD,EUR_USD,NZD_CAD,AUD_CAD,EUR_GBP,GBP_USD,CHF_JPY.log",
        "update_news_release_signals_allEUR_CAD,USD_CHF,EUR_CHF,AUD_USD,GBP_AUD,NZD_CHF,CAD_CHF.log",
        "update_news_release_signals_allEUR_JPY,GBP_CAD,NZD_JPY,CAD_JPY,GBP_CHF,NZD_USD,USD_JPY.log",
        "update_news_release_signals_allEUR_NZD,GBP_JPY,AUD_CHF,EUR_AUD,GBP_NZD,USD_CAD,AUD_JPY.log"]:

    fileHandle = open (file_name, "r" )
    line_list = fileHandle.readlines()
    fileHandle.close()

    last_time = 0
    for line in line_list[-100:]:

        ln = line[len("2018-09-21 12:23:05 "):]

        if ln.startswith("Finished"):

            local = datetime.datetime.strptime(line[:len("2018-09-21 12:23:05")], "%Y-%m-%d %H:%M:%S")

            local = local.replace(tzinfo=from_zone)

            # Convert time zone
            utc = local.astimezone(to_zone)

            last_time = calendar.timegm(utc.timetuple())

    if time.time() - last_time > 60 * 60 * 3:
        errors.append(file_name + " " + str((time.time() - last_time) / (60 * 60)))


    
file1 = open("/var/www/html/dashboard.html","w") 

for error in errors:
    file1.write("Error Update " +  error + "<br>")

file1.write("<hr>")
  
for index in range(len(image_names)):
    file1.write(image_names[len(image_names) - index - 1])

file1.close() #to change file access modes 
       


