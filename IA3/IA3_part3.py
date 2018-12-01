import numpy as np
from math import log
# import matplotlib.pyplot as plt
import csv
import math
import operator
from collections import OrderedDict
import time

class Node:
    def __init__(self, theda=None, depth=None, lchild=None, rchild=None, feature=0, label=None):
        self.lchild = lchild
        self.rchild = rchild
        self.depth = depth
        self.theda = theda
        self.feature = feature
        self.label = label

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
 data_left, data_right : [y, x1....,xn, index]
 '''
 left_pos_idx = np.where((data_left.T)[0]==1)[0]
 left_neg_idx = np.where((data_left.T)[0]==-1)[0]
 right_pos_idx = np.where((data_left.T)[0]==1)[0]
 right_neg_idx = np.where((data_left.T)[0]==-1)[0]

 count_right_pos = np.sum(D[right_pos_idx])
 count_right_neg = np.sum(D[right_neg_idx])
 count_left_pos = np.sum(D[left_pos_idx])
 count_left_neg = np.sum(D[left_neg_idx])

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
	for y in range(1,size_x):	
		x_array_sorted = x_array[x_array[:,y].argsort()]
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
				temp_feature = y
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
	return  theda, best_feature, left_neg, left_pos, right_neg, right_pos, left_array, right_array


def setlabel(pos, neg):
	if pos >= neg:
		return 1
	else:
		return -1


def create_node(root, depth, total_train,root_pos,root_neg, accur,size_x):

	theda, best_feature, left_neg, left_pos, right_neg, right_pos, left_array, right_array = best_B(total_train,root_pos,root_neg,size_x)
	
	root.feature = best_feature
	root.theda = theda
	root.depth = depth
	root.label = setlabel(root_pos, root_neg)
	a = []

	accur[root.depth+1] = accur.setdefault(root.depth+1, 0) + min(left_neg, left_pos) + min(right_neg, right_pos)

	if root.depth < max_depth:
		if left_neg != 0 and left_pos != 0:
			root.lchild = Node()
			root.lchild = create_node(root.lchild, depth+1, left_array,left_pos,left_neg, accur, size_x)
		else:
			root.lchild = Node()
			root.lchild.label = setlabel(left_pos, left_neg)
		
		if right_neg != 0 and right_pos != 0:
			root.rchild = Node()
			root.rchild = create_node(root.rchild, depth+1, right_array,right_pos,right_neg, accur, size_x)
		else:
			root.rchild = Node()
			root.rchild.label = setlabel(right_pos, right_neg)

	return root

# def compute_accur(accur, leng):
# 	print('============accur=================================')
# 	for key in accur:
# 		print('depth: ', key, " ;  accur:", 1-(accur[key]/leng))

def sort_cut(total_valid, feature, theda):
	x_array_sorted = total_valid[total_valid[:,feature].argsort()]
	x_array_sorted_t = np.transpose(x_array_sorted)

	split_point = (x_array_sorted_t[feature] < theda).sum()

	left_array = x_array_sorted[0:split_point]
	right_array = x_array_sorted[split_point:]

	count_left_neg,count_left_pos,count_right_neg, count_right_pos = split_data(left_array, right_array)
	
	return left_array, right_array, count_left_neg,count_left_pos,count_right_neg, count_right_pos


###############Adaboost#############################

def output_pred_y(total_data, root):
	# print("total_data.shape[1]: ", total_data.shape[0])
	pred_y = np.zeros(total_data.shape[0])

	find_leaf(total_data, root, pred_y)
	return pred_y

def find_leaf(total_data, root, pred_y):
	if root.lchild == None or root.rchild == None:
		idx_array = get_index(total_data)
		for idx in idx_array:
			pred_y[int(idx)] = root.label

	else:
		left_array, right_array, _,_,_,_ = sort_cut(total_data, root.feature, root.theda)

		find_leaf(left_array, root.lchild, pred_y)
		find_leaf(right_array, root.rchild, pred_y)

def get_index(data):
	trans_data = np.transpose(data)
	idx = trans_data[-1]
	return idx

def Cal_Error(data, root):
	'''
		Input: label of leaf node(predict), y_idx
		1. Build the tree
		2. caculate error
		Output: error
	'''
	predict_y = output_pred_y(data, tree.root)
	data_y = (data.T)[0]

	add = predict_y + data_y
	result = (add==0).sum()
	total_num = len(data)
	err = result/total_num
	acc = 1-err
	
	return err, acc

def alpha(data, root):
	error, accur = Cal_Error(data, tree.root)
	return ((1/2)*math.log((1-error)/error))

def ChangeDistribution(data, root):
	predict_y = output_pred_y(data,tree.root)
	data_y = (data.T)[0]
	alp = alpha(data, tree.root)
	D_value = 0
	D_1 = []
	for i in range(0, len(D)):
		if predict_y[i] != data_y[i]:
			D_value = D[i] * math.exp(alp)
			D_1.append(D_value)
		else:
			D_value = D[i] * math.exp(-alp)
			D_1.append(D_value)
	D_sum = sum(D_1)
	next_D = np.array(D_1)/D_sum
	return next_D





############Main Function############
#train.csv
y_array_train = y_data('pa3_train_reduced.csv')
x_array_train = x_data('pa3_train_reduced.csv')
total_train = original_data('pa3_train_reduced.csv')
#valid.csv
y_array_valid = y_data('pa3_valid_reduced.csv')
x_array_valid = x_data('pa3_valid_reduced.csv')
total_valid = original_data('pa3_valid_reduced.csv')

data_idx=[]
for i in range(0, 4888):
	data_idx.append([i])
total_train = np.append(total_train, data_idx, axis = 1)

vdata_idx=[]
for i in range(0, 1629):
	vdata_idx.append([i])
total_valid = np.append(total_valid, vdata_idx, axis = 1)


length = list(range(len(x_array_train)))
dic_data = list(zip(length, total_train))
tree = Create_Tree()
accur = dict()
max_depth = 9
D = np.repeat(1/4888, 4888)

pos_idx = np.where(y_array_valid==1)[0] 
root_pos = np.sum(D[pos_idx])
neg_idx = np.where(y_array_valid==-1)[0]
root_neg = np.sum(D[neg_idx])

size_x = total_train.shape[1]


create_node(tree.root, 0, total_train, root_pos, root_neg, accur,size_x-1)
print(Cal_Error(total_train, tree.root))
