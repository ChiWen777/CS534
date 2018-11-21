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
	# y_to_array = np.array(y)
	y_to_array = y
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


############Root U Value######################
def root_u(y):
	pos = y.count(1)
	nag = y.count(-1)
	u = 1 - pow((pos/(pos+nag)), 2) - pow((nag/(pos+nag)), 2)
	return u 




############Compute U value###################
def compute_u(x, y):
	return 1

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

print(root_u(y_array_train))

print(x_array_train)
# print(x_feature(x_array_train, 1))			#1 is feature



