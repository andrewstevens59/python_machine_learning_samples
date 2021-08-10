import smtplib

from email.mime.text import MIMEText
import requests


msg = MIMEText('Testing some Mailgun awesomness')
msg['Subject'] = "Hello"
msg['From']    = "Ukrainian Model: <andrewstevens@quantforextrading.com>"
msg['To']      = "callum.mcdonald@uqconnect.edu.au"

s = smtplib.SMTP('smtp.mailgun.org', 587)

s.login('postmaster@mail.quantforextrading.com', '8db90d867f8d32cf709c7929cd101d5a-e566273b-fc9dad8f')
s.sendmail(msg['From'], msg['To'], msg.as_string())
s.quit()


def send_simple_message():
	return requests.post(
		"https://api.mailgun.net/v3/quantforextrading/messages",
		auth=("api", "2768224616de45a1e3804bee6bc57346-e566273b-b7044a3b"),
		data={"from": "Excited User <mailgun@quantforextrading>",
			"to": ["andrew.stevens.591@gmail.com", "andrewstevens@quantforextrading.com"],
			"subject": "Hello",
			"text": "Testing some Mailgun awesomness!"})

send_simple_message()