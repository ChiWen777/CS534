import numpy as np
from math import log
# import matplotlib.pyplot as plt
import csv
import math
import operator
from collections import OrderedDict
import time

class Node:
    def __init__(self, theda=None, depth=None, lchild=None, rchild=None, feature=None):
        self.lchild = lchild
        self.rchild = rchild
        self.depth = depth
        self.theda = theda
        self.feature = feature

class Create_Tree:
    def __init__(self):
        self.root = Node()
 
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
		total_to_array = np.array(total)
	return total_to_array

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
 # left_y = []
 # right_y = []
 # for i in range(0, len(data_left)):
 #  left_y.append(data_left[i][1][0])

 # for i in range(0, len(data_right)):
 #  right_y.append(data_right[i][1][0])
 # print(data_left)
 left_y = np.transpose(data_left)
 right_y = np.transpose(data_right)



 count_right_pos = (right_y[0]==1).sum()
 count_right_neg = (right_y[0]==-1).sum()
 count_left_pos = (left_y[0]==1).sum()
 count_left_neg = (left_y[0]==-1).sum()
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

def best_B(x_array, pos, neg, size_x):
	theda, left_neg, left_pos, right_neg, right_pos, left_array, right_array = 0, 0, 0, 0, 0, None, None
	U_root = U_value(neg, pos)
	# U_root = 1
	best_b = 0
	temp_theda_index = [0,0]
	theda_index= [0,0]
	pre_y_value=0
	curr_y_value=0
	left_y =[]
	best_feature = 0
	temp_feature = 0
	temp_left_neg=0
	temp_left_pos=0
	temp_right_neg=0
	temp_right_pos=0
	count = 0
	# for y in range(1,101):
	for y in range(1,size_x):
		print("looking feature",y)
		# print(x_array[np.lexsort((x_array[:,0],x_array[:,y]))])
		# x_array_temp[np.lexsort((x_array[:,0],x_array[:,y]))]
		x_array_sorted = x_array[x_array[:,y].argsort()]
		# x_array_sorted = x_array[np.lexsort((x_array[:,0],x_array[:,y]))]
		# print(x_array_sorted)
		# x_array_sorted=sorted(x_array, key= lambda x: x[1][y])
		pre_y_value=0
		curr_y_value=0
		count =0
		for i in range(1,np.size(x_array,0)):
			# print(i)
			pre_y_value = curr_y_value
			# curr_y_value = x_array_sorted[i][1][0]
			curr_y_value = x_array_sorted[i][0]
			# print(curr_y_value,pre_y_value)
			if pre_y_value != curr_y_value:
				count +=1
				temp_theda = x_array_sorted [i][y]
				temp_feature = i
				temp_left_neg, temp_left_pos, temp_right_neg, temp_right_pos = split_data(x_array_sorted[0:i],x_array_sorted[i:])
				# print(temp_left_neg, temp_left_pos, temp_right_neg, temp_right_pos)
				if temp_left_neg==temp_left_pos==0 or temp_right_pos==temp_right_neg==0:
					temp_b=0
				else:
					temp_b = B_value(temp_left_neg, temp_left_pos, temp_right_pos, temp_right_neg, U_root )
				if temp_b > best_b:
					best_feature = temp_feature
					theda = temp_theda
					left_pos = temp_left_pos
					left_neg = temp_left_neg
					right_neg = temp_right_neg
					right_pos = temp_right_pos
					left_array = x_array_sorted[0:i]
					right_array = x_array_sorted[i:]
					best_b = temp_b
		print (best_b)
		print ("computation count in feature: ",count)
	return  theda, best_feature, left_neg, left_pos, right_neg, right_pos, left_array, right_array

def create_node(root, depth, total_train,root_pos,root_neg, accur,size_x):

	theda, best_feature, left_neg, left_pos, right_neg, right_pos, left_array, right_array = best_B(total_train,root_pos,root_neg,size_x)
	
	root.feature = best_feature
	root.theda = theda
	root.lchild = Node()
	root.rchild = Node()
	root.depth = depth

	accur[root.depth] = accur.setdefault(root.depth, 0) + min(left_neg, left_pos) + min(right_neg, right_pos)
	print('==================================================', root.depth)
	if root.depth < max_depth:
		if left_neg != 0 and left_pos != 0:
			root.lchild = create_node(root.lchild, depth+1, left_array,left_pos,left_neg, accur, size_x)
		
		if right_neg != 0 and right_pos != 0:
			root.rchild = create_node(root.rchild, depth+1, right_array,right_pos,right_neg, accur, size_x)
	
	return root

def compute_accur(accur, leng):
	print('============accur=================================')
	for key, value in enumerate(accur):
		print('depth: ', key, " ;  accur:", 1-(accur[key]/leng))


############Main Function############
#train.csv
y_array_train = y_data('pa3_train_reduced.csv')
x_array_train = x_data('pa3_train_reduced.csv')
total_train = original_data('pa3_train_reduced.csv')
#valid.csv
y_array_valid = y_data('pa3_valid_reduced.csv')
x_array_valid = x_data('pa3_valid_reduced.csv')
total_valid = original_data('pa3_valid_reduced.csv')

# dic = list(range(0, 4888))
# dic_data = dict(zip(dic, total_train))
# values = list(dic_data.values())
# print(values[0][1])


######### train ######################
length = list(range(len(x_array_train)))
# print(total_train)
dic_data = list(zip(length, total_train))

# d_sorted_by_value =sorted(dic_data, key= lambda x: x[1][3])
# print(d_sorted_by_value)
root_pos = list(y_array_train).count(1)
root_neg = list(y_array_train).count(-1)
size_x = total_train.shape[1]-1

tree = Create_Tree()
accur = dict()
tree.root.depth = 0
max_depth = 20
create_node(tree.root, 0, total_train,root_pos,root_neg, accur, size_x)
compute_accur(accur, len(length))