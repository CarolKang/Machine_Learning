from __future__ import division


import numpy as np
import scipy
import scipy.io as sio
import scipy.stats as stats
from scipy.stats import itemfreq
import sklearn.metrics as metrics
from math import log
import csv

"""
    Spam detection using random forest w/bagging.
    @author: Wenjing Kang
"""

data = sio.loadmat('hw5_data/spam_data/spam_data.mat')
all_train_data, all_train_labels = data['training_data'], data['training_labels']
all_train_labels = all_train_labels.reshape(all_train_labels.shape[1])

# Divide into training and validation (1000 samples) sets.
val_num = 1030

validation_idx = (np.random.choice(len(all_train_labels), val_num, replace=False)).reshape(val_num)
training_idx = np.delete(np.arange(len(all_train_labels)), validation_idx)
train_data, train_labels= all_train_data[training_idx], all_train_labels[training_idx]
validation_data, validation_labels = all_train_data[validation_idx], all_train_labels[validation_idx]
test_data = data['test_data']

class Node:
	def __init__(self, label_prob, split_rule = None, left_child = None, right_child = None, min_samples = 2):
		self.label_prob = label_prob
		self.split_rule = split_rule
		self.left_child = left_child
		self.right_child = right_child
		self.min_samples = min_samples
		self.label = None

	def set_label(self):
		if self.label == None:
			self.label = max(self.label_prob, key=lambda label : self.label_prob[label])
			self.prob = self.label_prob[self.label]
		return self.label

class BinaryDecisionTree:
	def __init__(self, leaf_entropy, max_depth = None, min_samples = 2, sampling_method = range):
		self.root = None
		self.leaf_entropy = leaf_entropy
		self.max_depth = max_depth
		self.min_samples =  min_samples 
		self.sampling_method = sampling_method
		# self.split_rules = []

	def impurity(self, left_label_hist, right_label_hist):
		left_freq = left_label_hist[:,1]
		right_freq = right_label_hist[:,1]
		left_total = np.sum(left_freq)
		right_total = np.sum(right_freq)
		total = left_total + right_total

		left_entropy = sum([(count/total) * log((count/total), 2) for count in left_freq if count != 0])
		right_entropy = sum([(count/total) * log((count/total), 2) for count in right_freq if count != 0])
		return -(left_total * left_entropy + right_total * right_entropy)/total

	def segmenter(self, data, labels):
		best_impurity = float('inf')
		best_split_rule = None
		best_left_i = None
		best_right_i = None

		for feature in self.sampling_method(data.shape[1]):
			feature_values = data[:, feature]
			for value in np.unique(feature_values):
				left_idx = np.nonzero(feature_values < value)[0]
				right_idx = np.nonzero(feature_values >= value)[0]
				if (len(left_idx) == 0 or len(right_idx) == 0):
					continue
				left_labels, right_labels = labels[left_idx], labels[right_idx]
				left_hist, right_hist = itemfreq(left_labels), itemfreq(right_labels)

				cur_impurity = self.impurity(left_hist, right_hist)
				if cur_impurity < best_impurity:
					best_impurity = cur_impurity
					best_split_rule = (feature, value)
					best_left_i = left_idx
					best_right_i = right_idx
		return best_split_rule, best_left_i, best_right_i


	def train(self, data, labels):
		self.root = self._grow_tree(data, labels, 0)

	def predict(self, data):
		node = self.root
		while (node.label == None):
			feature, threshold = node.split_rule
			if (data[feature] < threshold):
				node = node.left_child
			else:
				node = node.right_child
		# self.split_rules.append((node.split_rule))
		return node.label, node.prob

	def print_predict(self, data):
		node = self.root
		while (node.label == None):
			feature, threshold = node.split_rule
			if (data[feature] < threshold):
				node = node.left_child
				print(feature, " < ", threshold)
			else:
				node = node.right_child
				print(feature, " > ", threshold)
		# self.split_rules.append((node.split_rule))
		# return node.label, node.prob


	def _grow_tree(self, data, labels, cur_depth):
		# print(cur_depth)
		cur_hist = itemfreq(labels)
		total = sum(cur_hist[:,1])
		entropy = sum([-(count/total) * log((count/total), 2) for count in cur_hist[:,1] if count != 0])
		if (self.max_depth != None and cur_depth == self.max_depth) or \
				labels.size < self.min_samples or \
				np.unique(labels).size == 1 or \
				entropy < self.leaf_entropy:
				return self._generate_leaf_node(labels)
		split_rules, left_idx, right_idx = self.segmenter(data, labels)
		if split_rules == None:
			return self._generate_leaf_node(labels)
		left_data, left_labels = data[left_idx], labels[left_idx]
		right_data, right_labels = data[right_idx], labels[right_idx]
		return Node(self._get_labels_hist(labels), \
					split_rule = split_rules, \
					left_child = self._grow_tree(left_data, left_labels, cur_depth + 1),\
					right_child = self._grow_tree(right_data, right_labels, cur_depth + 1))




	def _generate_leaf_node(self, labels):
		leaf = Node(self._get_labels_hist(labels))
		leaf.set_label()
		return leaf

	def _get_labels_hist(self, labels):
		freq = itemfreq(labels)
		total = sum([count for label, count in freq])
		labels_freq = dict({label : count/total for label, count in freq})
		return labels_freq


# ============ RANDOM FOREST ============ #
class RandomForest:
	
	def __init__(self, num_trees, leaf_entropy, bagging_ratio = 1, max_depth=None, sampling_method = range):
		self.trees = []
		self.num_trees = num_trees
		self.max_depth = max_depth
		self.bagging_ratio = bagging_ratio
		self.sampling_method = sampling_method
		self.leaf_entropy = leaf_entropy

	def bagging(self, data, labels):
		sample_size = int(data.shape[0] * self.bagging_ratio)
		sample_idx = np.random.choice(data.shape[0], sample_size)
		return data[sample_idx], labels[sample_idx]

	def train(self, data, labels):
		for idx in range(self.num_trees):
			tree = BinaryDecisionTree(leaf_entropy = self.leaf_entropy , max_depth = self.max_depth, sampling_method = self.sampling_method)
			data_sample, labels_sample = self.bagging(data, labels)
			tree.train(data_sample, labels_sample)
			self.trees.append(tree)

	def predict(self, data):
		"""given a data point, outputs ensemble predictions"""
		predictions = {}
		for tree in self.trees:
			prediction, prob = tree.predict(data)
			if prediction in predictions:
				predictions[prediction] += prob
			else:
				predictions[prediction] = prob
		return max(predictions, key=predictions.get)

	def print_popular_root(self):
		root_split = {}
		for tree in self.trees:
			rootNode = tree.root
			feature, value = rootNode.split_rule
			if (feature, value) in root_split:
				root_split[(feature, value)] += 1
			else:
				root_split[(feature, value)] = 1
		max_key = max(root_split, key=root_split.get)
		print((training_idx[max_key[0]], max_key[1]), root_split[max_key])
		del root_split[max_key]
		max_key2 = max(root_split, key=root_split.get)
		print((training_idx[max_key2[0]], max_key2[1]), root_split[max_key2])
		del root_split[max_key2]
		max_key3 = max(root_split, key=root_split.get)
		print((training_idx[max_key3[0]], max_key3[1]), root_split[max_key3])
		del root_split[max_key3]



#Decision Tree
classifier = BinaryDecisionTree(0.2, max_depth = 9)
classifier.train(train_data, train_labels)
predictions = [classifier.predict(datum)[0] for datum in train_data]
train_score = metrics.accuracy_score(train_labels, predictions)
print("Decision Tree Training Accuracy: {0}".format(train_score))

predictions = [classifier.predict(datum)[0] for datum in validation_data]
validation_score = metrics.accuracy_score(validation_labels, predictions)
print("Decision Tree Validation Accuracy: {0}".format(validation_score))

import random
datum_idx = random.randint(0, train_data.shape[0] - 1)
print("Find the split rules for datum ", datum_idx)
classifier.print_predict(train_data[datum_idx])


# Ramdom Forest
classifier = RandomForest(20, 0.01, max_depth= 60, \
							sampling_method = lambda n : \
							np.random.choice(n, size=(int(np.sqrt(n))+1), replace=False))
classifier.train(train_data, train_labels)
predictions = [classifier.predict(datum) for datum in train_data]
train_score = metrics.accuracy_score(train_labels, predictions)
print("Random Forest Train accuracy: {0}".format(train_score))


predictions = [classifier.predict(datum) for datum in validation_data]
validation_score = metrics.accuracy_score(validation_labels, predictions)
print("Random Forest Validation Accuracy: {0}".format(validation_score))

classifier.print_popular_root()
# kaggle_predictions = [classifier.predict(datum) for datum in test_data]

# with open('wenjing_spam_forest.csv', 'w') as csvfile:
#       writer = csv.writer(csvfile)
#       writer.writerow(['Id','Category'])
#       for i in range(len(kaggle_predictions)):
#           writer.writerow([i+1, kaggle_predictions[i]])
