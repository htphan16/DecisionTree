from math import log2
import pandas as pd
import numpy as np
import os

def read_file(filename):
	return pd.read_csv(filename, sep='\t')

# def store_examples(df, num_examples):
# 	return df.sample(num_examples)

def entropy(examples):
	for i in examples.columns.tolist():
		if i == 'playtennis' or i == 'survived' or i == 'iscat':
			goal_att = i
	q = examples[examples[goal_att] == 'yes'][goal_att].count()/examples[goal_att].count()
	ent = -(q*log2(q) + (1-q)*log2(1-q))
	return ent

'''Function to calculate information gain to determine each attribute's importance'''
def importance(examples, attribute):
	for i in examples.columns.tolist():
		if i == 'playtennis' or i == 'survived' or i == 'iscat':
			goal_att = i
	size = examples.groupby([attribute]).size()
	remainder = 0
	for j in size.index.tolist():
		a = examples.groupby([attribute]).get_group(j).count()[goal_att]/examples.shape[0]
		try:
			b = examples.groupby([attribute, goal_att]).get_group((j, 'yes')).count()[goal_att]
		except KeyError:
			b = 0
		c = examples.groupby([attribute]).get_group(j).count()[goal_att]
		q = b/c
		if q == 0 or q == 1: 
			ent = 0
		else:
			ent = -(q*log2(q) + (1-q)*log2(1-q))
		remainder = remainder + a*ent
	importance = entropy(examples) - remainder
	return importance

class Leaf(object):
	def __init__(self):
		self.classfication = None

class Tree(object):
	def __init__(self, attribute):
		self.attribute = attribute
		self.label = None
		self.subtree = None

def plurality_value(examples):
	for i in examples.columns.tolist():
		if i == 'playtennis' or i == 'survived' or i == 'iscat':
			goal_att = i
	value_counts = examples[goal_att].value_counts()
	if value_counts['no'] >= value_counts['yes']:
		return 'no'
	else:
		return 'yes'

def build_tree(examples, attributes, parent_examples):
	for i in examples.columns.tolist():
		if i == 'playtennis' or i == 'survived' or i == 'iscat':
			goal_att = i
	value_counts = examples[goal_att].value_counts()
	leaf = Leaf()
	if examples.to_dict('records') == []:
		file = open('decision_tree.txt', 'a')
		print(plurality_value(parent_examples), file = file)
		file.close()
		leaf.classfication = plurality_value(parent_examples)
		# print('==========')
		return leaf
	elif 'yes' not in value_counts:
		file = open('decision_tree.txt', 'a')
		print('no', file = file)
		file.close()
		leaf.classfication = 'no'
		# print('==========')
		return leaf
	elif 'no' not in value_counts:
		file = open('decision_tree.txt', 'a')
		print('yes', file = file)
		file.close()
		# print('==========')
		leaf.classfication = 'yes'
		return leaf
	elif attributes == []:
		file = open('decision_tree.txt', 'a')
		print(plurality_value(examples), file = file)
		file.close()
		# print('==========')
		leaf.classfication = plurality_value(examples)
		return leaf
	else:
		imp = 0
		for a in attributes:
			imp = max(imp, importance(examples, a))
		for a in attributes:
			if importance(examples, a) == imp:
				att_to_split = a
		tree = Tree(att_to_split)
		for value in examples[att_to_split].unique():
			file = open('decision_tree.txt', 'a')
			print(att_to_split + " = " + value, file = file)
			# tree.label = att_to_split + " = " + value
			# print('From ' + tree.label + ' to')
			print('|||||', file = file)
			file.close()
			exs = examples[examples[att_to_split] == value]
			if att_to_split in attributes:
				attributes.remove(att_to_split)
			subtree = build_tree(exs, attributes, examples)
			tree.label = att_to_split + " = " + value
			tree.subtree = subtree
			file = open('decision_tree.txt', 'a')
			print('Back to ' + tree.attribute + ':', file = file)
			print('.....', file = file)
			file.close()
		return tree
	
def display_tree(filename):
	examples = read_file(filename)
	# examples = store_examples(df, 200)
	for i in examples.columns.tolist():
		if i == 'playtennis' or i == 'survived' or i == 'iscat':
			goal_att = i
	attributes = examples.columns.tolist()
	attributes.remove(goal_att)
	file = open('decision_tree.txt', 'w')
	build_tree(examples, attributes, examples)
	file.close()
	name = 'decision_tree_' + filename
	# os.rename('decision_tree.txt', name)
	read_file = open(name, 'r')
	tree_file = [line.strip('\n') for line in read_file] 
	num_nodes = 0
	for line in tree_file:
		if line == '|||||' or line == 'no' or line == 'yes':
			num_nodes = num_nodes + 1
	print('Number of nodes in the tree is: ')
	return num_nodes
	read_file.close()

def training(examples, example_index):
	for i in examples.columns.tolist():
		if i == 'playtennis' or i == 'survived' or i == 'iscat':
			goal_att = i
	attributes = examples.columns.tolist()
	attributes.remove(goal_att)
	pred_ex = examples.iloc[[example_index]].to_dict('records')
	d = {}
	pred_ex_list = []
	for a in attributes:
		d.update({a: importance(examples, a)})
	att_sorted = []
	sorted_d = sorted(d.items(), key=lambda x: x[1], reverse=True)
	# print(sorted_d)
	for item in sorted_d:
		att_sorted.append(item[0])
	# print(att_sorted)
	for att in att_sorted:
		pred_ex_list.append(att + " = " + pred_ex[0][att])
	# print(pred_ex)
	# print(pred_ex_list)
	file = open('decision_tree.txt', 'w')
	tree = build_tree(examples, attributes, examples)
	file.close()
	read_file = open('decision_tree.txt', 'r')
	tree_file = [line.strip('\n') for line in read_file] 
	# print(tree_file)
	for label in pred_ex_list:
		if label in tree_file:
			tree_file = tree_file[tree_file.index(label):]
			# print(tree_file)
			if tree_file[tree_file.index(label) + 2] == 'yes' or tree_file[tree_file.index(label) + 2] == 'no':
				return tree_file[tree_file.index(label) + 2] == pred_ex[0][goal_att]
	return 'no' == pred_ex[0][goal_att]
		# else:
		# 	return 'no' == pred_ex[0][goal_att]
	read_file.close()

def training_set_accuracy(filename):
	examples = read_file(filename)
	count = 0
	# print(leave_one_out_cross_validation(examples, 0))
	for i in range(len(examples)):
		print(training(examples, i))
		if training(examples, i) == True:
			count = count + 1
	print("Accuracy: ", 100*count/len(examples))

def leave_one_out_cross_validation(examples, example_index):
	for i in examples.columns.tolist():
		if i == 'playtennis' or i == 'survived' or i == 'iscat':
			goal_att = i
	attributes = examples.columns.tolist()
	attributes.remove(goal_att)
	pred_ex = examples.iloc[[example_index]].to_dict('records')
	exs1 = examples.iloc[0:example_index,:]
	exs2 = examples.iloc[example_index+1:,:]
	exs = pd.concat([exs1, exs2])
	d = {}
	pred_ex_list = []
	for key in pred_ex[0]:
		if pred_ex[0][key] not in exs[key].unique():
			return 'no' == pred_ex[0][goal_att]
	for a in attributes:
		d.update({a: importance(exs, a)})
	att_sorted = []
	sorted_d = sorted(d.items(), key=lambda x: x[1], reverse=True)
	# print(sorted_d)
	for item in sorted_d:
		att_sorted.append(item[0])
	# print(att_sorted)
	for att in att_sorted:
		pred_ex_list.append(att + " = " + pred_ex[0][att])
	# print(pred_ex)
	# print(pred_ex_list)
	file = open('decision_tree.txt', 'w')
	tree = build_tree(exs, attributes, exs)
	file.close()
	read_file = open('decision_tree.txt', 'r')
	tree_file = [line.strip('\n') for line in read_file] 
	# print(tree_file)
	for label in pred_ex_list:
		if label in tree_file:
			tree_file = tree_file[tree_file.index(label):]
			# print(tree_file)
			if tree_file[tree_file.index(label) + 2] == 'yes' or tree_file[tree_file.index(label) + 2] == 'no':
				return tree_file[tree_file.index(label) + 2] == pred_ex[0][goal_att]
	return 'no' == pred_ex[0][goal_att]
		# else:
		# 	return 'no' == pred_ex[0][goal_att]
	read_file.close()

def accuracy_testing(filename):
	examples = read_file(filename)
	count = 0
	# print(leave_one_out_cross_validation(examples, 0))
	for i in range(len(examples)):
		print(leave_one_out_cross_validation(examples, i))
		if leave_one_out_cross_validation(examples, i) == True:
			count = count + 1
	print("Accuracy: ", 100*count/len(examples))
