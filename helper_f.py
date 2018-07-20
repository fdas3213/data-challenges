import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns

def plot_freq(columns, label, df):
	for column in columns:
		freq_table = pd.crosstab(index = df[column], columns = df[label], normalize = 'index')
		plt.figure()
		plt.bar(x = freq_table.index, height = freq_table[1])
		plt.xlabel(column)

def plot_continuous(columns, label, df):
	for column in columns:
		print(column)
		hist = df[[column, label]].hist(by = column, bins = 30)
		plt.show()

def plot_classifier(X,y):
	classifiers = [RandomForestClassifier(), LogisticRegression()]

	log_cols = ['Classifier', 'Accuracy']
	log = pd.DataFrame(columns = log_cols)

	sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 0)

	acc_dict = {}
	for train_index, test_index in sss.split(X,y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		for clf in classifiers:
			name = clf.__class__.__name__
			clf.fit(X_train, y_train)
			train_predictions = clf.predict(X_test)
			acc = accuracy_score(y_test, train_predictions)
			acc_dict[name] = acc_dict.get(name, 0) + acc

	for clf in acc_dict:
		acc_dict[clf] /= 10.0
		log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns = log_cols)
		log = log.append(log_entry)

	print(log)

	plt.xlbale('Accuracy')
	plt.title('Classifier Accuracy')

	sns.set_color_codes('muted')
	sns.barplit(x = 'Accuracy', y = 'Classifier', data = log, color = 'b')