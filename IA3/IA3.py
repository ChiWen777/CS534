import numpy as np
from math import log
import matplotlib.pyplot as plt
import csv
import math
import operator


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


############Root U Value######################
def root_u(y):
	pos = y.count(1)
	nag = y.count(-1)
	u = 1 - pow((pos/(pos+nag)), 2) - pow((nag/(pos+nag)), 2)
	return u 




############Split data###################
def split_data(y_index, t, t_index):
	'''
		y_index: list
		t : int (theta)
		t_index: [x, y]
		y: original y
	'''

	a = x_feature(x_array_train) 
	y = y_array_train
	x = a[t_index[1]]					#which column
	x_value = x[y_index]				#x's value(which row)
	y_value = y[y_index]				#y's value(which row)


	right_index, left_index = [], []
	left_y, right_y = {}, {}

	####split data######
	for v in range(0, len(y_index)):
		if x_value[v] >= t:
			right_index.append(y_index[v])
			right_y[v] = y_value[v]
		else:
			left_index.append(y_index[v])
			left_y[v] = y_value[v]


	right_y_value = right_y.values()
	left_y_value = left_y.values()
	count_right_pos = list(right_y_value).count(1)
	count_right_neg = list(right_y_value).count(-1)
	count_left_pos = list(left_y_value).count(1)
	count_left_neg = list(left_y_value).count(-1)
	return count_left_neg,count_left_pos,count_right_neg, count_right_pos, right_index, left_index



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
#valid.csv
y_array_valid = y_data('pa3_valid_reduced.csv')
x_array_valid = x_data('pa3_valid_reduced.csv')

# print(list(y_array_train).count(1))


dic = list(range(0, 4888))
com = list(zip(y_array_train, x_array_train))
dic_data = dict(zip(dic, com))
dic_v = list(dic_data.values())
a = list(com[0])
b = list(a[1])
print(b[3])
