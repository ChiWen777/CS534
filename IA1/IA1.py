import numpy as np
import pandas as pd
import csv
import os
import copy
import matplotlib.pyplot as plt


##part1
pwd = os.getcwd()
train = pd.read_csv('PA1_train.csv', sep=',',header=None)
train = train.values
test = pd.read_csv('PA1_test.csv', sep=',',header=None)
test = test.values
dev = pd.read_csv('PA1_dev.csv', sep=',',header=None)
dev = dev.values
raw_train_data = np.zeros((10000, 22))  ## take out id and price 
raw_test_data = np.zeros((6000, 22))  ## take out id 
raw_dev_data = np.zeros((5597, 22))  ## take out id and price 
normalized_train_data = np.zeros((10000, 22))  ## take out id and price 
normalized_test_data = np.zeros((6000, 22))  ## take out id 
normalized_dev_data = np.zeros((5597, 22))  ## take out id and price 
y_train_data = np.zeros((10000, ))
y_dev_data = np.zeros((5597, ))
# learning_list = [pow(10, 0),pow(10, -1),pow(10, -2),pow(10, -3),pow(10, -4),pow(10, -5),pow(10, -6),pow(10, -7)]

learning = pow(10, -5)
normalg_list = list()



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
                raw_train_data[:,count_col] = ea_date_data.reshape((10000,))
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_train_data[:, 0] = cut_head_data
            raw_train_data[:,0] = cut_head_data.reshape((10000,))
            count_col += 1
        elif ea_col == 1:
            pass
        elif ea_col == 21:
            cut_head_data = cut_head_data.astype(float)
            y_train_data = cut_head_data
        else:
            cut_head_data = cut_head_data.astype(float)
            raw_train_data[:,count_col] = cut_head_data.reshape((10000,))
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
                raw_test_data[:,count_col] = ea_date_data.reshape((6000,))
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_test_data[:, 0] = cut_head_data
            raw_test_data[:,0] = cut_head_data.reshape((6000,))
            count_col += 1
        elif ea_col == 1:
            pass
        else:
            cut_head_data = cut_head_data.astype(float)
            raw_test_data[:,count_col] = cut_head_data.reshape((6000,))
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
                raw_dev_data[:,count_col] = ea_date_data.reshape((5597,))
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_dev_data[:, 0] = cut_head_data
            raw_dev_data[:,0] = cut_head_data.reshape((5597,))
            count_col += 1
        elif ea_col == 1:
            pass
        elif ea_col == 21:
            cut_head_data = cut_head_data.astype(float)
            y_dev_data = cut_head_data
        else:
            cut_head_data = cut_head_data.astype(float)
            raw_dev_data[:,0] = cut_head_data.reshape((5597,))
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm)
            count_col += 1
            
    return y_train_data, y_dev_data




def grad(w, x, y, lamda):   
    """
    The gradient of the linear regression with l2 regularization cost function
    x:input dataset
    y:output dataset
    lamda:regularization factor
    """
    sse = 0
    sum_up = 0
    N = x.shape[0]      #we need to know how many data in each column(How many rows)

    for i in range(0, N):
        sse += (y[i] - np.dot(w, x[i]))**(2)
        sum_up += 2 * (np.dot(w, x[i]) - y[i]) * x[i] + 2 * lamda * w
    print("sse:",sse)
    normalg_list.append(sse)
    return sum_up


def grad_descent (x, y, learning):
    """
    The grad_descent function of different learning rate and fixed lamda
    w: weight
    learning: learning rate
    converage: converage limit value
    """ 

    w = np.zeros(22)
    converage=40
    i=0
    for runs in range(1000000):
        i=i+1
        gradient = grad(w, x, y, 0)
        w = w - (learning * gradient)
        normalg= np.linalg.norm(gradient)
        print("normalg: ", normalg)
        #print("gradient: ", gradient)
        if np.isinf(normalg):
            print(normalg_list)
            break
        #normalg_list.append(gradient)
        if normalg <= converage:
            print("normalg <= converage!!!")
            break
  
    print("total run: ",i)
    print("w: ",w )
    return w


def test_y_value(w, x):
    '''
        This function is for finding y value for test. file
        w: Best w value
        x: test. file without price column
        y: use y value from train data or validation data
    '''

    pred_y = np.array([])           #store pred_value

    for i in x:
        value = np.dot(w, i)    
        pred_y = np.append(pred_y, value)

    return pred_y


def cross_comparison_dev(w, true_dev_y):
    pred_dev_y = test_y_value(w, normalized_dev_data)
    sum_difference_y = float()
    for (ea_true_dev_y, ea_pred_dev_y )in zip(true_dev_y, pred_dev_y):
        difference_y = abs(ea_true_dev_y - ea_pred_dev_y)
        sum_difference_y += difference_y

    print(sum_difference_y)

    
if __name__ == "__main__":
    y_train_data, y_dev_data = process_columns()
    grad_descent(normalized_train_data, y_train_data, learning)
    plt.plot(normalg_list)
    plt.savefig(pwd+"/pic.png")
    # plt.show()
    del normalg_list[:]

    # cross_comparison_dev( bill_input_w, y_dev_data)
