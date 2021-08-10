import os
from selenium import webdriver

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located



import subprocess

process = subprocess.Popen(
    ['WinSCP.com', '/ini=nul', 'C:\\diskpartscript.txt',
     'open ftp://Administrator:Passwor81@66.45.235.203', 'get *.txt', 'exit'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for line in iter(process.stdout.readline, b''):  # replace b'' with '' for Python 2
    print(line.decode().rstrip())

sys.exit(0)

DRIVER_PATH = '/Users/andrewstevens/Downloads/chromedriver_2'

chrome_options = Options()
chrome_options.add_argument('--disable-gpu')
#chrome_options.add_argument('--headless')
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=chrome_options)


driver.get('https://www.forexfactory.com/calendar.php?day=oct21.2020')
print (driver.page_source.encode('utf-8').decode('latin-1'))
