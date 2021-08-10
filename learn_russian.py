from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import random as rand
from selenium.webdriver.firefox.options import Options

options = Options()
options.add_argument('--headless')

driver = webdriver.Firefox(options=options, executable_path='/Users/andrewstevens/Downloads/geckodriver')

text = """
very slim person
very fat person
i will discover the truth
I will search in the woods
i will kill the person who killed my brother
Death is a part of life
i advice you to study well
Very ugly person
Gloomy weather today
how often do you exercise
what are your hobbies
What is your favourite animal
have you ever been on a boat
how often do you clean your apartment
i dont want to exercise anymore
i cannot fly because i am not a bird
do you find me attractive
are you interested in politics
do you like living here
please take-off shoes
i don't want to sleep now
do you want to join the army
From what made this car
i will climb the stairs
repeat many times and you will get better
that person will attack you
you need a hospital
i want to live an easy life
run fast or you will die
Do you like to swim in the ocean
what do you do when you have free time
i will look for you in the library
modern technology is cool
very rainy weather today
Speak more slowly please
can you help me please
i am getting tired, can i please go home
I want to do something else in my life.
i don't know how to make this happen
you are enemies of japan
i need to be able to communicate with people
you can only enjoy the art and teach people how to dance
thats all - i have nothing else to say
you will be regarded as enemies
dont look at me
lets get you settled into you apartment
you can stand in line next to me
apparently he grew up wealthy
He expected to become an officer but was denied
This news has not been received well
None of this explains his erratic behaviour 
They were looking for answers
i will correct you
Please turn left
who is the olympic athlete
thats our revenge
he thought i was better than i am
Why do you make me hit you
people around me
i will use it to fix my car
im alive and well
As incorrectly announced by your government
heard my voice
Keep my guns good condition so I can go hunting
you are responsible for this problem
he does not have enough experience for this job
we are both strong
i will return
it is necessary to have respect
what is better than desert
 will run to the shops
hopefully i will find the correct answer
i would like to jump on the trampoline
it my preference
I robbed the bank
you have strange definition of warm
anyone who will not work
no one knows you're here
i will carry your bag
how did you approach the person
I manufacture drugs
keep going you're almost done
all these precautions did not help
you must be polite
my friends have real names
its best you do what you are told
do not shout loudly
pickup the box
if he drops it shoot him
why are you laughing at me
i think we are related
Would you consider visit my country
what activity do you have planned for today
I finally understood what they were thinking
I recently acquired a new helicopter
I don’t have any interaction with him 
I did not realize it was so soons
until i found out it was making me fat
in my country a nurse also offers sex
she does not look like her photo
i came to moscow and agreed to go
i will probably see you there
i spent all day in the garden
"""

lines = text.split("\n")

driver.get('https://www.forexfactory.com/calendar.php')
html = driver.page_source
print (html)
sys.exit(0)

driver.get("https://translate.yandex.com/?lang=en-ru&text=hello")
time.sleep(10)

while True:
	index = rand.randint(0, len(lines) - 1)
	driver.get("https://translate.yandex.com/?lang=en-ru&text=" + lines[index])
	word_count = len(lines[index].split(" ")) * 2
	time.sleep(word_count)

