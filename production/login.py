from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import time

from flask import Flask
from flask import request

driver = webdriver.Firefox(executable_path = '/usr/local/bin/geckodriver')

driver.get("https://www.instagram.com/")

#login
time.sleep(5)
username = driver.find_element_by_css_selector("input[name='username']")
password = driver.find_element_by_css_selector("input[name='password']")
username.clear()
password.clear()
username.send_keys("xx")
password.send_keys("Password81")
login = driver.find_element_by_css_selector("button[type='submit']").click()

#save your login info?
time.sleep(10)
notnow = driver.find_element_by_xpath("//button[contains(text(), 'Not Now')]").click()
#turn on notif
time.sleep(10)
notnow2 = driver.find_element_by_xpath("//button[contains(text(), 'Not Now')]").click()

#searchbox
time.sleep(5)
searchbox = driver.find_element_by_css_selector("input[placeholder='Search']")
searchbox.clear()
searchbox.send_keys("host.py")
time.sleep(5)
searchbox.send_keys(Keys.ENTER)
time.sleep(5)
searchbox.send_keys(Keys.ENTER)

app = Flask(__name__)
@app.route("/")
def hello():
 	handle = request.args.get('handle')

	driver.get("https://www.instagram.com/" + handle + "/?__a=1")
	html = driver.page_source
	return html


if __name__ == "__main__":



    app.run()



