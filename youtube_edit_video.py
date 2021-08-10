from moviepy.editor import *
from moviepy.editor import VideoFileClip, concatenate_videoclips


import numpy as np
from moviepy.editor import *
from moviepy.video.tools.segmenting import findObjects
import random

# WE CREATE THE TEXT THAT IS GOING TO MOVE, WE CENTER IT.


rotMatrix = lambda a: np.array( [[np.cos(a),np.sin(a)], 
                                 [-np.sin(a),np.cos(a)]] )

def vortex(screenpos,i,nletters):
    d = lambda t : 1.0/(0.3+t**8) #damping
    a = i*np.pi/ nletters # angle of the movement
    v = rotMatrix(a).dot([-1,0])
    if i%2 : v[1] = -v[1]
    return lambda t: screenpos+400*d(t)*rotMatrix(0.5*d(t)*a).dot(v)
    
def cascade(screenpos,i,nletters):
    v = np.array([0,-1])
    d = lambda t : 1 if t<0 else abs(np.sinc(t)/(1+t**4))
    return lambda t: screenpos+v*400*d(t-0.15*i)

def arrive(screenpos,i,nletters):
    v = np.array([-1,0])
    d = lambda t : max(0, 3-3*t)
    return lambda t: screenpos-400*v*d(t-0.2*i)
    
def vortexout(screenpos,i,nletters):
    d = lambda t : max(0,t) #damping
    a = i*np.pi/ nletters # angle of the movement
    v = rotMatrix(a).dot([-1,0])
    if i%2 : v[1] = -v[1]
    return lambda t: screenpos+400*d(t-0.1*i)*rotMatrix(-0.2*d(t)*a).dot(v)

def moveLetters(letters, funcpos):
    return [ letter.set_pos(funcpos(letter.screenpos,i,len(letters)))
              for i,letter in enumerate(letters)]

def create_video_description(count):

	txtClip = TextClip('Number ' + str(count),color='white', font="Amiri-Bold",
	                   kerning = 5, fontsize=100)
	cvc = CompositeVideoClip( [txtClip.set_pos('center')])
	letters = findObjects(cvc) # a list of ImageClips

	clips = [ CompositeVideoClip( moveLetters(letters,funcpos)).subclip(0,5)
          for funcpos in [vortex, cascade, arrive, vortexout] ]

	return concatenate_videoclips(clips)


for video_count in range(100):
	query = "swimsuit models"
	mypath = "/Users/andrewstevens/Downloads/youtube_videos/" + query + "/"

	descriptions = [
	"Trade Forex Profitably",
	"Quit Your Job - Learn To Trade",
	"Trade and Travel Around The World",
	"Use News Releases To Trade",
	"Gain A Statistical Edge In Forex Trading",
	"Use Fundamental Economic News",
	"Economic Forecasting",
	"Trade Forex Using Sophisticated Machine Learning",
	"Quit Your Job - Live The Life You Want",
	"Earn Money While You Travel",
	"Trade and Invest For Your Future",
	"Hate Your Job - So Quit",
	"News Releases Move Markets",
	"Earn Money While You Sleep",
	"The Smart Way To Earn An Income",
	]

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	videos = []
	for file in onlyfiles:
		if ".mp4" not in file:
			continue

		start_point = random.uniform(0, 1)
		clip = VideoFileClip(mypath + file)
		video1 = clip.subclip(int(clip.duration * start_point),int(clip.duration * start_point) + 10).resize( (720,460))

		txt_clip1 = ( TextClip(file.replace(".mp4", ""),fontsize=20,color='white')
	             .margin(top=25, opacity=0)
	             .set_position(("center","top"))
	             .set_duration(10) )

		txt_clip2 = ( TextClip("www.quantforextrading.com",fontsize=20,color='yellow')
	             .margin(bottom=45, opacity=0)
	             .set_position(("center","bottom"))
	             .set_duration(10) )

		txt_clip3 = ( TextClip(descriptions[random.randint(0,len(descriptions)-1)],fontsize=20,color='white')
	             .margin(bottom=25, opacity=0)
	             .set_position(("center","bottom"))
	             .set_duration(10) )

		video1 = CompositeVideoClip([video1, txt_clip1, txt_clip2, txt_clip3]) # Overlay text on video

		videos.append(video1)


	video = concatenate_videoclips(videos)
	video.write_videofile("/Users/andrewstevens/Downloads/youtube_videos/best_of_hottest " + query + "_" + str(video_count) + ".mp4", fps=None,
	        codec="libx264",
	        audio_codec="aac",
	        bitrate=None,
	        audio=True,
	        audio_fps=44100,
	        preset='medium',
	        audio_nbytes=4,
	        audio_bitrate=None,
	        audio_bufsize=2000,
	        rewrite_audio=True,
	        verbose=True,
	        threads=None,
	        ffmpeg_params=None)  


