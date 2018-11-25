import numpy as np
from math import log
# import matplotlib.pyplot as plt
import csv
import math
import operator
from collections import OrderedDict
import time


##########Read File###########
def original_data(filename):
	file = open(filename , 'r')
	add_o = []
	total = []
	for row in file:
		tmp = []						#each row creates a list
		for x in row.split(','):		
			tmp.append(float(x.strip()))		
		add_o.append(tmp)
	a = np.array(add_o)
	for x_v in a:
		str_xv = str(x_v[0])
		if str_xv == '5.0':
			p_str = str_xv.replace('5.0', '-1')
			x_v[0] = float(p_str)
		else:
			n_str = str_xv.replace('3.0', '1')
			x_v[0] = float(n_str)
		total.append(x_v)
	return total

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
	# y_to_array = y
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
		add_x.append(tmp[1:])
	x_to_array = np.array(add_x)			# x list convert to x array
	
	return x_to_array

############get each feature##################
def x_feature(x):
	'''
	'''
	x_T = x.T
	return x_T

############Split data###################
def split_data(data_left, data_right):
 '''
 data_left, data_right : [index, y, x1....,xn]
 '''
 left_y = []
 right_y = []
 for i in range(0, len(data_left)):
  left_y.append(data_left[i][1][0])

 for i in range(0, len(data_right)):
  right_y.append(data_right[i][1][0])

 count_right_pos = right_y.count(1)
 count_right_neg = right_y.count(-1)
 count_left_pos = left_y.count(1)
 count_left_neg = left_y.count(-1)
 return count_left_neg,count_left_pos,count_right_neg, count_right_pos

# ############Root U Value######################
def U_value(neg,pos):

	u = 1 - pow((pos/(pos+neg)), 2) - pow((neg/(pos+neg)), 2)
	return u 


def B_value(left_neg,left_pos, right_pos, right_neg, U_root):
	pb_l= (left_pos+left_neg)/(right_pos+right_neg+left_pos+left_neg)
	pb_r= (right_pos+right_neg)/(right_pos+right_neg+left_pos+left_neg)
	B_value = U_root - pb_l*U_value(left_neg,left_pos) - pb_r*U_value(right_neg,right_pos)
	return B_value

def best_B(x_array, neg, pos):
	U_root = U_value(neg, pos)
	best_b = 0
	temp_theda_index = [0,0]
	theda_index= [0,0]
	pre_y_value=0
	curr_y_value=-1
	left_y =[]
	temp_left_neg=0
	temp_left_pos=0
	temp_right_neg=0
	temp_right_pos=0
	for y in range(1,101):
		print("looking feaure",y)
		time_a = time.clock()
		x_array_sorted=sorted(x_array, key= lambda x: x[1][y])
		for i in range(1,(len(x_array))):
			pre_y_value = curr_y_value
			curr_y_value = x_array_sorted[i][1][0]
			if pre_y_value != curr_y_value:
				temp_left_neg, temp_left_pos, temp_right_neg, temp_right_pos = split_data(x_array_sorted[0:i-1],x_array_sorted[i:-1])
				if temp_left_neg==temp_left_pos==0 or temp_right_pos==temp_right_neg==0:
					temp_b=0
				else:
					temp_b = B_value(temp_left_neg, temp_left_pos, temp_right_pos, temp_right_neg, U_root )
				if temp_b > best_b:
					left_pos = temp_left_pos
					left_neg = temp_left_neg
					right_neg = temp_right_neg
					right_pos = temp_right_pos
					left_array = x_array_sorted[0:i-1]
					right_array = x_array_sorted[i:-1]
					best_b = temp_b
	return  left_neg, left_pos, right_neg, right_pos, left_array, right_array



############Compute U value###################

# def Split_samples(self, x_to_array, feature):
# 	'''
		
# 	'''
# 	ret = {}
# 	for x in x_to_array:
# 		val = x[feature]
# 		ret.setdefault(val, [])
# 		ret[val].append(x)
# 	return ret

# 	# file = open(filename , 'r')
# 	# add_x = []
# 	# x = []
# 	# a = csv.reader(file)
# 	# for row in a :
# 	# 	for i in range(2):
# 	# 		add_x.append(row[i])
# 	# 	x.append(add_x)
# 	# x_to_array = np.array(x)
# 	# return x_to_array




# ###########Split data#######################
# def Split_data(self, x, level=0):
# 	'''
# 	'''
# 	if stop_now(x):
# 		return x[0][-1]

# 	#split data
# 	feature = self.get_feature(x, level)
# 	subsets = self.Split_samples(s, feature)

# 	return {key: self.Split_data(subset, level+1) for key, subset in subsets.item()}


# #########
# def stop_now(self, x):
# 	'''
# 	'''
# 	labels = [d[-1] for d in x]

# 	return len(set(labels)) <=1

# ##########
# def get_feature(self, x, level):
# 	'''
# 	'''
# 	return level

# class DecisionTree(object):
# 	'''
# 	'''
# 	def __init__(self, x):
# 		super(DecisionTree, self).__init__()
# 		self.root = self.Split_data(x)


# tree = DecisionTree(x_array_train)
# print(tree.root)



############Main Function############
#train.csv
y_array_train = y_data('pa3_train_reduced.csv')
x_array_train = x_data('pa3_train_reduced.csv')
total_train = original_data('pa3_train_reduced.csv')
#valid.csv
y_array_valid = y_data('pa3_valid_reduced.csv')
x_array_valid = x_data('pa3_valid_reduced.csv')


# dic = list(range(0, 4888))
# dic_data = dict(zip(dic, total_train))
# values = list(dic_data.values())
# print(values[0][1])

length = list(range(len(x_array_train)))
# print(total_train)
dic_data = list(zip(length, total_train))

# d_sorted_by_value =sorted(dic_data, key= lambda x: x[1][3])
# print(d_sorted_by_value)
root_pos = list(y_array_train).count(1)
root_neg = list(y_array_train).count(-1)


print(best_B(dic_data,root_pos,root_neg))

# print(dic_data)
##############data for sorted#################
# values.sort(key=operator.itemgetter(1))\
# d_sorted_by_value =sorted(dic_data.items(), key= lambda x: x[1][3])
# dic.sort(key=operator.itemgetter(1))
# print(values)
# print(d_sorted_by_value)
# for key,value in sorted(dic_data.items(), key= lambda x: x[1][3]):
# 	print (key)
