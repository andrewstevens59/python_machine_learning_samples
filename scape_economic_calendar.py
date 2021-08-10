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

def get_calendar_data(date):

	url='https://www.forexfactory.com/calendar.php?day=' + date
	#Create a handle, page, to handle the contents of the website
	page = requests.get(url)

	time_zone_inc = page.content.find("Calendar Time Zone: GMT ")
	time_zone_inc += len("Calendar Time Zone: GMT ")
	time_component = page.content[time_zone_inc:time_zone_inc+5]
	time_offset = int(time_component.split(' ')[0]) * 60 * 60
	time_offset += (60 * 60)

	#Store the contents of the website under doc
	doc = lh.fromstring(page.content)
	#Parse data that are stored between <tr>..</tr> of HTML
	tr_elements = doc.xpath('//tr')




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
				if 'class="better"' in html:
					better_worse = "B"
				if 'class="worse"' in html:
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
					print data, "Space"
					continue

				if actual == None:
					actual = data
					print data, "Actual"
					continue

				if forecast == None:
					forecast = data
					print data, "Forecast"
					continue

				if previous == None:
					previous = data
					print previous, "Previous"
					print description, "description"

					utc = datetime.datetime.strptime(date + " " + curr_time, "%b%d.%Y %I:%M%p")
					timestamp = calendar.timegm(utc.timetuple())
					timestamp -= time_offset

					print (str(timestamp) + "," + currency + "," + description + "," + actual + "," + forecast + "," + previous + "," + str(impact)) 
					trade_logger.info(str(timestamp) + "," + currency + "," + description + "," + actual + "," + forecast + "," + previous + "," + str(impact)) 
					continue

			if pattern.match(data):
				curr_time = data

			if data in currencies:
				print date, curr_time, data
				found_currency = True
				currency = data


from datetime import timedelta, date
import datetime as dt

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

def daterange(start_date, end_date):
	for n in range(int ((end_date - start_date).days)):
		yield start_date + timedelta(n)


import os
for year in range(2015, 2020):
	if os.path.exists("/Users/andrewstevens/Downloads/economic_calendar/calendar_" + str(year) + ".txt"):
	  os.remove("/Users/andrewstevens/Downloads/economic_calendar/calendar_" + str(year) + ".txt")

	trade_logger = setup_logger('first_logger', "/Users/andrewstevens/Downloads/economic_calendar/calendar_" + str(year) + ".txt")

	start_date = date(year, 1, 1)
	end_date = date(year, 12, 31)
	for single_date in daterange(start_date, end_date):
		get_calendar_data(single_date.strftime("%b%d.%Y").lower())

	handlers = trade_logger.handlers[:]
	for handler in handlers:
	    handler.close()
	    trade_logger.removeHandler(handler)
