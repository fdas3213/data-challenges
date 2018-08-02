import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from datetime import datetime

"""use this function to visualize the relationship between
a categorical variable and the target variable. Check out distribution of 
one class of categorical variable vs. target variable
"""
def plot_freq(columns, label, df, ylabel, rotation = False):
	for column in columns:
		freq_table = pd.crosstab(index = df[column], columns = df[label], normalize = 'index')
		plt.figure()
		freq_table.plot.bar()
		plt.xlabel(column)
		plt.ylabel(ylabel)
		if rotation:
			plt.xticks(rotation = 45)

"""use this function to visualize continuous variable vs.
one class of target variable
"""
def plot_continuous(columns, label, df):
	for column in columns:
		print(column)
		df[column].hist(bins = 30)
		hist = df[[column, label]].hist(by = label, bins = 30)
		plt.show()

"""
check out mean value of a continuous variable vs. one class of 
target variable
"""
def plot_mean_value(columns, label, df):
	for column in columns:
		plt.figure()
		new_df = df.groupby(label)[column].mean()
		new_df.plot.bar()
		plt.ylabel("average " + column)
        
"""
user histogram to plot frequency of target varibale
"""
def plot_target(label, df):
	df[label].hist()
	plt.xlabel(label)
	plt.ylabel('frequency')


def plot_feature_importance(df, label, clf, degree):
	feature_columns = df.loc[:, df.columns != label].columns.values
	imp_df = pd.DataFrame({"features":feature_columns, "importance": clf.feature_importances_})
	imp_df = imp_df.sort_values('importance').reset_index(drop = True)
	plt.title("Feature importance")
	plt.bar(x = imp_df['features'], height = imp_df['importance'])
	plt.xticks(rotation = degree)
    

def t_test(df, group_label, label):
	group_a = df[df[group_label] == 0][label]
	group_b = df[df[group_label] == 1][label]
	#calculate mean and std
	var_a = group_a.var(ddof = 1)
	var_b = group_b.var(ddof = 1)
	denom = np.sqrt((var_a/len(group_a)) + (var_b/len(group_b)))
	t_score = (group_a.mean() - group_b.mean()) / denom
	return t_score

"""consider adding SVM and GB to this list"""
def plot_classifier(X,y):
	classifiers = [RandomForestClassifier(), LogisticRegression(), GradientBoostingClassifier()]

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

	plt.xlabel('Accuracy')
	plt.title('Classifier Accuracy')

	sns.set_color_codes('muted')
	sns.barplit(x = 'Accuracy', y = 'Classifier', data = log, color = 'b')