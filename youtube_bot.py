#importing the module 
from pytube import YouTube 

'''

#YouTube("https://www.youtube.com/watch?v=LffLqUwlY2Y").streams.first().download('/Users/andrewstevens/Downloads/')

import urllib2
from bs4 import BeautifulSoup

textToSearch = 'slava'
query = urllib2.quote(textToSearch)
url = "https://www.youtube.com/results?search_query=" + query
response = urllib2.urlopen(url)
html = response.read()
soup = BeautifulSoup(html, 'html.parser')
for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
    print('https://www.youtube.com' + vid['href'])

    response = urllib2.urlopen('https://www.youtube.com' + vid['href'])
    html = response.read()
    print html

'''


import csv
import googleapiclient.discovery

def most_popular(yt, **kwargs):
    popular = yt.videos().list(chart='mostPopular', part='snippet', **kwargs).execute()
    for video in popular['items']:
        yield video['snippet']

yt = googleapiclient.discovery.build('youtube', 'v3', developerKey=…)
with open('YouTube Trending Titles on 12-30-18.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Title', 'Description'])
    csv_writer.writerows(
        [snip['title'], snip['description']]
        for snip in most_popular(yt, maxResults=20, regionCode=…)
    )