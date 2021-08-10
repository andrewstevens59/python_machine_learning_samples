

x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

y1 = [81,63,32,54,88,32,37,33,16,98,5,74,7,97,56,50,81,88,97,29]

x2= [1,4.8,8.6,12.399999999999999,16.2,20]

y2 = [78.5,33.25,57.75,65.75,27.5,55]
	


import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(x1, y1, label="original")
plt.plot(x2, y2, label="approximate")

plt.show()



