import numpy as np
import pandas as pd
import csv
import os

train = pd.read_csv('PA1_train.csv', sep=',',header=None)
train = train.as_matrix()

"""
w: weight
learning: learning rate
lam: lamda for gradient computation
converage: converage limit value
"""	
def grad_descent(DATA, y, learning, lam, converage):

	w = np.zeros(45)
       
	for runs in range(1000000):
		gradient = grad (w, DATA[runs,:], y, lam)
		w = w - (learning * gradient)
		if runs % 1000 == 0:
			print ("w: ", w)
		if g <= converage:
			break
		if runs >= 200000:
			break

	return w          

"""
    The gradient of the linear regression with l2 regularization cost function
    x:input dataset
    y:output dataset
    lamda:regularization factor
    
"""
def grad(w, x, y, lamda): 	
	
    sum_up = 0
    N = x.shape[0]		#we need to know how many data in each column(How many rows)

    for i in range(0, N):
        sum_up = 2 * (np.dot(w, x[i]) - y[i]) * y[i] + 2 * lamda * w
    return sum_up



'''
	The regularization of different lamda values
	x:input dataset
    y:output dataset
    lamda:regularization factor
    rate:learning rat
'''
def diff_lamda(x, y, rate, lamda):
	
	w = 1	#initial w
	rate = 			#fixed rate

	# gradient descent algorithm with different lamda
	lamda_array = [0.001, 0.01, 0.1, 0, 1, 10, 100]
	for lamda in lamda_array:
		E = grad(w, x, y, lamda)
		w = w - ( rate * E)
		print(w, E)
    return w

    
