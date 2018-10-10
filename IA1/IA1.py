import numpy as np
import pandas as pd
import csv
import os

train = pd.read_csv('PA1_train.csv', sep=',',header=None)
train = train.as_matrix()
test = pd.read_csv('PA1_test.csv', sep=',',header=None)
test = test.as_matrix()

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
		if gradient <= converage:
			break
		if runs >= 200000:
			break

	return w          

def split_date(cut_head_data):

    split_date_data = copy.deepcopy(cut_head_data)

    sd_data = np.zeros((10000,3))
        
    for idx_r, ea_date_str in enumerate(split_date_data):
        data_features = ea_date_str.split("/")
        for idx in range(0,3):
            sd_data[idx_r,idx] = data_features[idx-1]
        idx_r += 1
    h_data_set = np.hsplit(sd_data,3)
    return h_data_set

def add_in_arrays(ea_col, data, min_array, max_array):
    """
    Add the max and the min in an array.
    """
    max_array[ea_col] = np.max(data)
    min_array[ea_col] = np.min(data)

def norm_data(ea_col, cut_head_data, min_array, max_array):
    """
    Normalize data.
    """
    new_data = (cut_head_data - min_array[ea_col]) / (max_array[ea_col] - min_array[ea_col])
    print(new_data)
    print('=====')
#     normalized_data[:, ea_col] = new_data

def process_columns():
    """
    Process both test.csv and train.csv 's columns and normalize them
    """

    # Run through every col in train.csv
    normalized_data = np.zeros((10000, 23))
    min_array = np.zeros((train.shape[1]+1,))
    max_array = np.zeros((train.shape[1]+1,))
    # print(mean_array.shape)

    for ea_col in range(train.shape[1]-1):
        orig_data = train[:,ea_col]
        
        cut_head_data = copy.deepcopy(orig_data)
        cut_head_data = cut_head_data[1:]
        if ea_col == 2:
            date_data = split_date(cut_head_data)
            for ea_date_data in date_data:
                add_in_arrays(ea_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, ea_date_data, min_array, max_array)
        elif ea_col == 0:
            pass
        else:
            cut_head_data = cut_head_data.astype(float)
            add_in_arrays(ea_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, cut_head_data, min_array, max_array)

    ##########################################################################

    # Run through every col in test.csv
    normalized_data = np.zeros((10000, 23))
    min_array = np.zeros((test.shape[1]+2,))
    max_array = np.zeros((test.shape[1]+2,))

    for ea_col in range(test.shape[1]):
        orig_data = test[:,ea_col]
        
        cut_head_data = copy.deepcopy(orig_data)
        cut_head_data = cut_head_data[1:]
        if ea_col == 2:
            date_data = split_date(cut_head_data)
            for ea_date_data in date_data:
                add_in_arrays(ea_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, ea_date_data, min_array, max_array)
        elif ea_col == 0:
            pass
        else:
            cut_head_data = cut_head_data.astype(float)
            add_in_arrays(ea_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, cut_head_data, min_array, max_array)




def grad_descent():
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

    
