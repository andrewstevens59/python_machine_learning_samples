import pycurl
from StringIO import StringIO

def download_calendar(time_frame):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	#31536000
	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/labs/v1/calendar?period=" + str(time_frame))


	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()

	title_map = {}

	import json
	j = json.loads(response_value)

	import re
	import sys
	regex = re.compile(".*?\((.*?)\)")


	features = []
	for i in range(len(j)):
		item = j[i]


		if 'previous' not in item:
			continue
	 
		forecast = item['previous']

		if re.search('[a-zA-Z]', forecast):
			continue
			
		'''
		if 'forecast' in item:
			forecast = item['forecast']
		'''

		if 'actual' not in item:
			continue

		if re.search('[a-zA-Z]', item['actual']):
			continue

		impact = 0
		if 'impact' in item:
			impact = item['impact']

		features.append([item['currency'], impact, item['actual'], forecast, item['timestamp'], item['region']])

	return features


