import requests
import lxml.html as lh
import pandas as pd
import logging
import time
import datetime
from dateutil import tz
import calendar
import re
import os.path
import pickle


pattern = re.compile("(1[012]|[1-9]):[0-5][0-9](\\s)?(?i)(am|pm)")



def get_interest_data():

	with open('interest_data.txt', 'r') as myfile:
   		interest_data = myfile.read().replace('\n', '')

	#Parse data that are stored between <tr>..</tr> of HTML
	doc = lh.fromstring(interest_data)
	tr_elements = doc.xpath('//tr')

	from_zone = tz.tzlocal()
	to_zone = tz.tzutc()


	currencies = ["GBP", "USD", "AUD", "CAD", "NZD", "JPY", "CHF", "EUR"]

	curr_time = None

	data_rows = []
	#Since out first row is the header, data is stored on the second row onwards
	for j in range(0,len(tr_elements)):
	    #T is our j'th row
	    T=tr_elements[j]

	    data_columns = []
	    #Iterate through each element of the row
	    for t in T.iterchildren():
	        data=t.text_content().strip()

	        data_columns.append(data)


	    if len(data_columns) == 4:
			data_rows.append(data_columns)

			#Mon Aug 25 09:05:19 2014

	

	final_interest_data = []
	for data_row in data_rows[1:]:
		data_columns = data_row

		toks = data_columns[-1].split(" ")
		local = datetime.datetime.strptime(data_columns[-1], "%a %b %d %H:%M:%S %Y")

		data_columns[1] = float(data_columns[1])
		data_columns[2] = float(data_columns[2])

		local = local.replace(tzinfo=from_zone)

		# Convert time zone
		utc = local.astimezone(to_zone)

		data_columns[-1] = calendar.timegm(utc.timetuple())
		final_interest_data.append(data_columns + [int(toks[-1])])


	interest_pd = pd.DataFrame(final_interest_data, columns = data_rows[0] + ["YEAR"])

	print interest_pd

	pickle.dump(interest_pd, open("/Users/callummc/interest_data.pickle", 'wb'))


from datetime import timedelta, date
import datetime as dt

get_interest_data()

