import numpy as np
import pandas as pd
import csv
import os
import copy


train = pd.read_csv('PA1_train.csv', sep=',',header=None)
train = train.as_matrix()
test = pd.read_csv('PA1_test.csv', sep=',',header=None)
test = test.as_matrix()
dev = pd.read_csv('PA1_dev.csv', sep=',',header=None)
dev = dev.as_matrix()
normalized_train_data = np.zeros((10000, 22))  ## take out id and price 
normalized_test_data = np.zeros((6000, 22))  ## take out id 
normalized_dev_data = np.zeros((5597, 22))  ## take out id and price 
y_train_data = np.zeros((10000, ))
y_dev_data = np.zeros((5597, ))
          


def split_date(cut_head_data, whichForm):
    split_date_data = copy.deepcopy(cut_head_data)
    if whichForm == 'train':
        sd_data = np.zeros((10000,3))
    if whichForm == 'test':
        sd_data = np.zeros((6000,3))
    if whichForm == 'dev':
        sd_data = np.zeros((5597,3))
        
    for idx_r, ea_date_str in enumerate(split_date_data):
        data_features = ea_date_str.split("/")
        for idx in range(0,3):
            sd_data[idx_r,idx] = data_features[idx-1]
        idx_r += 1
    h_data_set = np.hsplit(sd_data,3)
    return h_data_set

def add_in_arrays(count_col, data, min_array, max_array):
    """
    Add the max and the min in an array.
    """
    max_array[count_col] = np.max(data)
    min_array[count_col] = np.min(data)

def norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm):
    """
    Normalize data.
    """
    new_data = (cut_head_data - min_array[count_col]) / (max_array[count_col] - min_array[count_col])
    
    if whichForm == 'train':
        if ea_col == 2:
            new_data =  new_data.reshape((10000,))
        normalized_train_data[:,count_col] = new_data

    if whichForm == 'test':
        if ea_col == 2:
            new_data =  new_data.reshape((6000,))
        normalized_test_data[:,count_col] = new_data
        
    if whichForm == 'dev':
        if ea_col == 2:
            new_data =  new_data.reshape((5597,))
        normalized_dev_data[:,count_col] = new_data


def process_columns():
    """
    Process both test.csv and train.csv 's columns and normalize them
    The final normalized data will store in normalized_train_data and normalized_test_data (without 'id' and 'price' columns )


    """
    
    count_col = 0
    
    # Run through every col in train.csv
    whichForm = 'train'
    min_array = np.zeros((train.shape[1],))
    max_array = np.zeros((train.shape[1],))

    for ea_col in range(train.shape[1]):
        
        orig_data = train[:,ea_col]
        
        cut_head_data = copy.deepcopy(orig_data)
        cut_head_data = cut_head_data[1:]
        
        if ea_col == 2:
            date_data = split_date(cut_head_data, whichForm)
            for ea_date_data in date_data:
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_train_data[:, 0] = cut_head_data
            count_col += 1
        elif ea_col == 1:
            pass
        elif ea_col == 21:
            y_train_data = cut_head_data
        else:
            cut_head_data = cut_head_data.astype(float)
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm)
            count_col += 1

    ##########################################################################
    
    count_col = 0
    
    # Run through every col in test.csv
    whichForm = 'test'
    min_array = np.zeros((test.shape[1]+1,))
    max_array = np.zeros((test.shape[1]+1,))

    for ea_col in range(test.shape[1]):
        orig_data = test[:,ea_col]
        
        cut_head_data = copy.deepcopy(orig_data)
        cut_head_data = cut_head_data[1:]
        
        if ea_col == 2:
            date_data = split_date(cut_head_data, whichForm)
            for ea_date_data in date_data:
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_test_data[:, 0] = cut_head_data
            count_col += 1
        elif ea_col == 1:
            pass
        else:
            cut_head_data = cut_head_data.astype(float)
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm)
            count_col += 1


    ##########################################################################
    
    count_col = 0
    
    # Run through every col in dev.csv
    whichForm = 'dev'
    min_array = np.zeros((dev.shape[1],))
    max_array = np.zeros((dev.shape[1],))

    for ea_col in range(dev.shape[1]):
        
        orig_data = dev[:,ea_col]
        
        cut_head_data = copy.deepcopy(orig_data)
        cut_head_data = cut_head_data[1:]
        
        if ea_col == 2:
            date_data = split_date(cut_head_data, whichForm)
            for ea_date_data in date_data:
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_dev_data[:, 0] = cut_head_data
            count_col += 1
        elif ea_col == 1:
            pass
        elif ea_col == 21:
            y_dev_data = cut_head_data
        else:
            cut_head_data = cut_head_data.astype(float)
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm)
            count_col += 1
            
    return y_train_data, y_dev_data

"""
    The gradient of the linear regression with l2 regularization cost function
    x:input dataset
    y:output dataset
    lamda:regularization factor
    
"""
def grad(w, x, y, lamda):   
    
    sum_up = 0
    N = x.shape[0]      #we need to know how many data in each column(How many rows)

    for i in range(0, N):
        sum_up = 2 * (np.dot(w, x[i]) - y[i]) * y[i] + 2 * lamda * w
    return sum_up

"""
The grad_descent function of different learning rate and fixed lamda
w: weight
learning: learning rate
converage: converage limit value
""" 
def grad_descent (x, y, learning):

    w = np.zeros(20)
    converage=0.5

    for runs in range(1000000):
        gradient = grad(w, normalized_train_data, y_train_data, 0)
        w = w - (learning * gradient)
        normalg= np.linalg.norm(gradient)
        if runs % 1000 == 0:
            print ("w: ", w)
        if normalg <= converage:
            break
        if runs >= 200000:
            break

    return normalg, w


'''
    The regularization of different lamda values and fixed learning rate
    x:input dataset
    y:output dataset
    lamda:regularization factor
    rate:learning rat
'''
def diff_lamda(x, y, lamda):
    
    w = np.zeros(20)   #initial w
    rate =  #fixed rate
    converage=0.5

    # gradient descent algorithm with different lamda
    lamda_array = [0.001, 0.01, 0.1, 0, 1, 10, 100]
    for lamda in lamda_array:
        for runs in range(1000000):
            E = grad(w, normalized_train_data, y_train_data, lamda)
            w = w - ( rate * E)
            if normalg <= converage:
                break
            
    return normalg, w

    
if __name__ == "__main__":
    y_train_data, y_dev_data = process_columns()
