import numpy as np
from math import log
import matplotlib.pyplot as plt
import csv
import math


##########Read File###########
def y_data(filename):
	'''
		This function is for adding y value in list
	'''
	file = open(filename , 'r')
	add_y = []
	y = []
	a = csv.reader(file)
	for row in a : 
		add_y.append(row[0])
	for iy in add_y:
		integer = int(iy)
		if integer == 3:
			y.append(float(1))
		else:
			y.append(float(-1))
	y_to_array = np.array(y)
	return y_to_array

def x_data(filename):
	'''
		This function is for x value without y value
	'''
	file = open(filename, 'r') 
	add_x = []
	for row in file:
		tmp = []						#each row creates a list
		for x in row.split(','):		
			tmp.append(float(x.strip()))
		tmp.append(float(1))		
		add_x.append(tmp[1:])
	x_to_array = np.array(add_x)			# x list convert to x array

	return x_to_array


#############Perceptron##############
def Perceptron(x, y):
	'''
		
	'''
	w = np.zeros(len(x[0]))		#number of values in Train.csv and Valid.csv
	it = 1					#iterate
	N = x.shape[0] 	
	accuracy = 0
	while it < 16:
		count = 0
		u = 0 					#initial x dot w.T
		for i in range(0, N):

			u = np.dot(x[i], w.T)

			if (y[i]*u) <= 0:
				count += 1
				w = w + y[i]*x[i]

		accuracy = 1-(count)/N
		print(accuracy)		
		it = it + 1
	return w


###########Test value################
# def test_value():


############Main Function############
#train.csv
y_array = y_data('pa2_train.csv')
x_array = x_data('pa2_train.csv')
#valid.csv
# y_array = y_data('pa2_valid.csv')
# x_array = x_data('pa2_valid.csv')

# Perceptron(x_array, y_array)
Kernel_Perceptron(x_array, y_array)


