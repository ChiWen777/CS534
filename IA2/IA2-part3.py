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




#############Kernel Perceptron##############
def Kernel_Perceptron(x, y):
	'''
		X = N*F, a = N*1, y = N*1 
	'''
	p_value = [1, 2, 3, 7, 15]
	N = x.shape[0]					#4888
	a = np.zeros(x.shape[0])		
	

	it = 1						#iterate

	for p in p_value:
		while it < 16:
			count = 0
			for i  in range(0, N):
				u = 0 
				for j in range(0, N):
					kp = (1 + np.dot(x[j], x[i].T)) ** p
					u = u + kp*a[j]*y[j] 
				if (y[i]*u) <= 0:
					count += 1
					a[i] += 1
			accuracy = 1- count/N
			print("p: ", p)	
			print(accuracy)
			it = it + 1
	return u







############Main Function############
#train.csv
t_y_array = y_data('pa2_train.csv')
t_x_array = x_data('pa2_train.csv')
#valid.csv
y_array = y_data('pa2_valid.csv')
x_array = x_data('pa2_valid.csv')

print("======train======")
train_u = Kernel_Perceptron(t_x_array, t_y_array)
print("======valid======")
train_u = Kernel_Perceptron(x_array, y_array)


