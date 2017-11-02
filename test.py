import numpy   as np 
import scipy.misc # to visualize only  
from matplotlib import pyplot as plt

def retrieve_sample(n):
	f = open('train_x.csv')
	out = open('sample.csv', 'w')

	for i in range(0,n):
		l = f.readline()
		out.write(l)

	print(len(l), l)
	

def showtest(n):
	x = np.loadtxt("sample.csv", delimiter=",") # load from text 
	x = x.reshape(-1, 64, 64) # reshape 
	for i in range(n):
		plt.imshow(x[i]) # to visualize only 
		plt.show()

retrieve_sample(100)
showtest()
