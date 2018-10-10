import numpy as np
import pandas as pd
import csv
import os

train = pd.read_csv('PA1_train.csv', sep=',',header=None)
train = train.as_matrix()


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

    
