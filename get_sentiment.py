import unicodecsv as csv
from selenium  import webdriver
import re
import urllib
from selenium.webdriver.chrome.options import Options
import time
import os

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import pandas as pd

url_final = []

def get_trade_bias(pair, is_buy):

    df = pd.read_csv('sentiment.csv')
    df = df[df['Symbol'] == pair]

    if len(df) == 0:
        return 1.0

    try:

        if is_buy:
            if float(df['long_perc']) > 50:
                return float(100 - abs(50 - float(df['long_perc']))) / 100
            else:
                return float(100 + abs(50 - float(df['long_perc']))) / 100
        else:
            if float(df['short_perc']) > 50:
                return float(100 - abs(50 - float(df['short_perc']))) / 100
            else:
                return float(100 + abs(50 - float(df['short_perc']))) / 100
    except:
        return 1.0

def get_url():

    url = 'https://www.dailyfx.com/sentiment'

    chrome_options = Options() 
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument('--dns-prefetch-disable')
    chrome_options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(executable_path=os.path.abspath("/usr/lib/chromium-browser/chromedriver"),   chrome_options=chrome_options) 
    driver.get(url)

    temp = driver.find_elements_by_class_name('dfx-technicalSentimentCard--pageView')
    delay = 3 # seconds
    try:
        myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'dfx-technicalSentimentCard__netShortContainer')))
        print "Page is ready!"
    except TimeoutException:
        print "Loading took too much time!"

    try:
        myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'dfx-technicalSentimentCard__netLongContainer')))
        print "Page is ready!"
    except TimeoutException:
        print "Loading took too much time!"

    try:
        myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'dfx-rateDetail__percentageInfoText')))
        print "Page is ready!"
    except TimeoutException:
        print "Loading took too much time!"

    time.sleep(5)
    temp = driver.find_elements_by_class_name('dfx-technicalSentimentCard--pageView')
    time.sleep(5)
    temp = driver.find_elements_by_class_name('dfx-technicalSentimentCard--pageView')

    index = 0
    for element in temp:
        text = element.text
        symbol = element.find_element_by_class_name('dfx-technicalSentimentCard__pairAndSignal').find_element_by_class_name('dfx-technicalSentimentCard__pair--link').text
        signal = element.find_element_by_class_name('dfx-technicalSentimentCard__pairAndSignal').find_element_by_class_name('dfx-technicalSentimentCard__signal').text
        net_long = element.find_element_by_class_name('dfx-technicalSentimentCard__netLongContainer').find_element_by_class_name('dfx-rateDetail__percentageInfoText').get_attribute('data-value')
        net_short = element.find_element_by_class_name('dfx-technicalSentimentCard__netShortContainer').find_element_by_class_name('dfx-rateDetail__percentageInfoText').get_attribute('data-value')
        url_final.append({'Symbol': symbol.replace("/", "_"),'Signal': signal,'long_perc' : net_long,'short_perc': net_short})

        # print('Symbol =',symbol,'Signal = ', signal)
        # print('Netlong = ',net_long + '%', 'Netshort = ',net_short + '%')

    with open('/root/sentiment.csv', 'wb') as csvFile:
        fields = ['Symbol', 'Signal', 'long_perc', 'short_perc']
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(url_final)


if __name__ == "__main__":
  get_url()

