import numpy as np
from math import log
# import matplotlib.pyplot as plt
import csv


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
        tmp = []                        #each row creates a list
        for x in row.split(','):        
            tmp.append(float(x.strip()))        
        add_x.append(tmp[1:])
    x_to_array = np.array(add_x)            # x list convert to x array

    return x_to_array


#############Average Perceptron##############
def Avg_Perceptron(x, y):
    '''
        
    '''
    dummy = np.ones(x.shape[0]).reshape(4888, 1)
    x = np.hstack((x,dummy))
    w = np.zeros(x.shape[1])
    avg_w = np.zeros(x.shape[1])
    s = 0
    c = 0
    itr = 0                    
    u = 0

    N = x.shape[0]        
    accuracy = 0

    while itr < 16:
        c = 0
        acc_t = 0
        for i in range(0, N):
            u = np.dot(x[i], w.T)
            if (y[i]*u) <= 0:
                if s+c > 0:
                    avg_w = (np.dot(s, avg_w) + np.dot(c, w))/(s+c)
                s += c
                w = w + y[i]*x[i]
                c = 0
            else:
                c += 1
                
        for j in range(0, N):
            u = np.dot(x[j], avg_w.T)
            if (y[j]*u) > 0:
                acc_t += 1
                
        accuracy = acc_t/x.shape[0]
        print(accuracy)   
            
        itr = itr + 1
        
    if c > 0:
        avg_w = (np.dot(s, avg_w) + np.dot(c, w))/(s+c)
                
    return w


############Main Function############

y_array = y_data('pa2_train.csv')
x_array = x_data('pa2_train.csv')
Avg_Perceptron(x_array, y_array)