
import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from random import *
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import time
import os.path

from sklearn import datasets, linear_model, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import numpy as np
from sklearn.decomposition import PCA

import pickle

import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from nltk.stem.porter import *

class PredictionMicroService:

	trained_model = None
	predictors = None
	mean_predictor_values = None
	stemmer = PorterStemmer()
	sale_id_lookup_table = None
	product_lookid_lookup_table = None


	# Creates a 2D PCA grid and calculates average sales at each grid point
	def createPCAFeatures(self, X_train, y_train, X_all, grid_size=10):

		pca = PCA(n_components=2)
		X = pca.fit_transform(X_all)

		pca1 = [float(row[0]) for row in X]
		pca2 = [float(row[1]) for row in X]

		pca1 = pd.qcut(pca1, [float(0.1) * i for i in range(11)], retbins=False, labels=False)
		pca2 = pd.qcut(pca2, [float(0.1) * i for i in range(11)], retbins=False, labels=False)
		X = zip(pca1, pca2)

		import bisect 

		label_offset = 0
		grid_rate = {}
		grid_num = {}
		for (a, b) in X:

		    grid_pos = min(9, int(a)) + (min(9, int(b)) * 10)

		    if grid_pos not in grid_rate:
		        grid_rate[grid_pos] = 0
		        grid_num[grid_pos] = 0

		    grid_rate[grid_pos] = grid_rate[grid_pos] + y_train[label_offset]
		    grid_num[grid_pos] = grid_num[grid_pos] + 1
		    label_offset = label_offset + 1

		    if label_offset >= len(y_train):
		    	break

		avg_label = np.mean(y_train)

		lookup_table = {}
		lookup_table[-1] = avg_label

		pca_features = []
		for (a, b) in X:
		    grid_pos = min(9, int(a)) + (min(9, int(b)) * 10)

		    net_rate = grid_rate[grid_pos] / grid_num[grid_pos]

		    if grid_num[grid_pos] < 5:
		        net_rate = avg_label
		    

		    lookup_table[grid_pos] = net_rate
		    pca_features.append(net_rate)

		return pca_features

	# This takes a long time to run so left out of the exmaple API
	def tune_model_parameters(self, X_train, y_train, X_test, y_test): 

		tuned_parameters = {'n_estimators': [50, 75, 100], 'max_depth': [4, 5, 6], 'min_samples_split': [2],
		      'learning_rate': [0.01], 'loss': ['ls']}

		print("# Tuning hyper-parameters for r2")
		print()

		clf = GridSearchCV(ensemble.GradientBoostingRegressor(), tuned_parameters, cv=2, scoring='r2')
		clf.fit(X_train, y_train)

		return clf


	def train_model_on_predictors(self, predictors, X_train, y_train, X_test, y_test):

		params = {'n_estimators': 50, 'max_depth': 4, 'min_samples_split': 3,
	          'learning_rate': 0.01, 'loss': 'ls'}
		clf = ensemble.GradientBoostingRegressor(**params)

		clf.fit(X_train, y_train)

		rmse = sqrt(mean_squared_error(y_test, clf.predict(X_test)))
		r2 = r2_score(y_test, clf.predict(X_test))

		print "##################### GradientBoostingRegressor Accuracy #####################"
		print("RMSE: ", rmse, "R2", r2)

		rf = RandomForestRegressor(max_depth=4, random_state=0)

		rf.fit(X_train, y_train)

		rmse = sqrt(mean_squared_error(y_test, rf.predict(X_test)))
		r2 = r2_score(y_test, rf.predict(X_test))

		print "##################### RandomForestRegressor Accuracy #####################"
		print("RMSE: ", rmse, "R2", r2)

		regression = linear_model.LinearRegression()
		regression.fit(X_train, y_train)

		rmse = sqrt(mean_squared_error(y_test, regression.predict(X_test)))
		r2 = r2_score(y_test, regression.predict(X_test))

		print "##################### LinearRegression Accuracy #####################"
		print("RMSE: ", rmse, "R2", r2)

		percentile = np.percentile(y_train, 95)

		print "95th percentile", percentile



		###################### tuned parameters ######################

		#tune_parameters(X_train, y_train, X_test, y_test)


		return rf 

	def quadratic_expansion(self, predictors, X):

		predictor_len = len(predictors)

		for i in range(predictor_len):
			for j in range(predictor_len - i):

				column_name = str(i) + "_" + str(j)
				X[column_name] = X[predictors[i]] - X[predictors[i + j]]
				predictors.append(column_name)

		return predictors, X

	# This is an alternative to one hot encoding
	def empirical_encoding(self, df, train_df, column_name):

		#train_df is used to stop information leakage

		avg_units_sold_df = train_df.groupby([column_name]).apply(lambda x: float(x['dependent_variable'].sum()) / max(x['dependent_variable'].count(), 1)).reset_index(name='emp_' + column_name)

		return pd.merge(df, avg_units_sold_df, on=column_name), avg_units_sold_df

	def stem_function(self, x): 
		words = x.split(" ") 
		stem_words = [self.stemmer.stem(word) for word in words]

		return ' '.join(stem_words)


	def stem_words(self, df, column_name):

		
		df[column_name] = df[column_name].apply(self.stem_function)

		return df



	def train_model(self, sales_df, feature_importance_filter):

	
		train_test_offset = int(len(sales_df) * 0.9)

		print float(len(sales_df['sale_id'].unique())) / len(sales_df)

		sys.exit(0)

		sales_df = self.stem_words(sales_df, 'shoe_type')

		sales_df, self.sale_id_lookup_table = self.empirical_encoding(sales_df, sales_df[:train_test_offset], 'sale_id')

		sales_df, self.product_lookid_lookup_table = self.empirical_encoding(sales_df, sales_df[:train_test_offset], 'product_look_id')

		sales_df.drop([u'sale_id', u'id', u'product_look_id', u'sale_start_time', u'sale_end_time', u'sale_type_key',  u'num_units_sold'], axis=1, inplace=True)

		categorical_fields = ['brand_id', 'return_policy_id', 'material_name', 'color', 'country_of_origin', 'shoe_type']



		sales_df = pd.get_dummies(sales_df, columns = categorical_fields)

		predictors = sales_df.columns.tolist()

		predictors.remove('dependent_variable')

		# Use this if your data has missing values etc
		#sales_df.fillna(sales_df.mean(), inplace=True)

		X = sales_df[predictors]
		y = sales_df['dependent_variable']

		#sales_df['emp_PCA'] = self.createPCAFeatures(X[:offset], y[:offset], X)
		#predictors.append('emp_PCA')

		X = sales_df[predictors]

		X_train, y_train = X[:train_test_offset], y[:train_test_offset]
		X_test, y_test = X[train_test_offset:], y[train_test_offset:]


		clf = self.train_model_on_predictors(predictors, X_train, y_train, X_test, y_test)

		feature_importance = clf.feature_importances_

		reduced_predictors = []
		print "##################### Feature Importance Weights #####################"
		for feature_weight, feature_name in zip(feature_importance, predictors):
			if feature_weight > feature_importance_filter: 
				print feature_weight, feature_name
				reduced_predictors.append(feature_name)


		# Sometimes this can help, but GradientBoostingRegressor is good at capturing interaction 
		#reduced_predictors, X = self.quadratic_expansion(reduced_predictors, X)

		X_train = X[reduced_predictors][:train_test_offset]
		X_test = X[reduced_predictors][train_test_offset:]

		# Create a reduced/light weight version of sthe model to remove low feature importance predictors
		clf = self.train_model_on_predictors(reduced_predictors, X_train, y_train, X_test, y_test)

		# this is used to test online and offline prediction match - would need more testing in practice
		print "first predictor value - used to check offline and online results match", clf.predict(X_train.iloc[0])

		self.trained_model = clf
		self.predictors = reduced_predictors
		self.mean_predictor_values = sales_df.mean()

	def model_sales_rate(self, feature_importance_filter):
		sales_df = pd.read_csv("Gilt_datascience_exercise.csv", delimiter=',')

	

		sales_df['sale_start_time'] = sales_df['sale_start_time'].str.slice(0, 19).apply(lambda d: time.mktime(datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timetuple()))
		sales_df['sale_end_time'] = sales_df['sale_end_time'].str.slice(0, 19).apply(lambda d: time.mktime(datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timetuple()))

		# sale duration in days
		duration = (sales_df['sale_end_time'] - sales_df['sale_start_time']) / (60 * 60 * 24)

		sales_df['dependent_variable'] = sales_df['num_units_sold'].astype('float') / duration.astype('float')

		self.train_model(sales_df, feature_importance_filter)

	def model_total_sales(self, feature_importance_filter):

		sales_df = pd.read_csv("Gilt_datascience_exercise.csv", delimiter=',')

		sales_df['dependent_variable'] = sales_df['num_units_sold']

		self.train_model(sales_df, feature_importance_filter)


	def save_model(self, path): 
		pickle.dump(self.trained_model, open(path + '/' + 'trained_model','w'))
		pickle.dump(self.predictors, open(path + '/' + 'predictors','w'))
		pickle.dump(self.mean_predictor_values, open(path + '/' + 'mean_predictor_values','w'))

		pickle.dump(self.sale_id_lookup_table, open(path + '/' + 'sale_id_lookup_table','w'))
		pickle.dump(self.product_lookid_lookup_table, open(path + '/' + 'product_lookid_lookup_table','w'))


	def load_model(self, path): 
		self.trained_model = pickle.load(open(path + '/' + 'trained_model','r'))
		self.predictors = pickle.load(open(path + '/' + 'predictors','r'))
		self.mean_predictor_values = pickle.load(open(path + '/' + 'mean_predictor_values','r'))

		self.sale_id_lookup_table = pickle.load(open(path + '/' + 'sale_id_lookup_table','r'))
		self.product_lookid_lookup_table = pickle.load(open(path + '/' + 'product_lookid_lookup_table','r'))

		self.product_lookid_lookup_table.set_index('product_look_id', inplace=True)
		self.sale_id_lookup_table.set_index('sale_id', inplace=True)

	def derive_predictors(self, X_test, columns):

		listofzeros = [0] * len(self.predictors)

		for column in columns:
			combined_column_value = str(column) + "_" + str(X_test[column])

			# if the value was not seen during one hot encoding than it is not 
			# assigned a corresponding boolean value - default to all zeros
			if combined_column_value in self.predictors:
				index = self.predictors.index(combined_column_value)
				listofzeros[index] = 1

			empirical_lookup_value = "emp_" + str(column)
			## Look for empirical CTR value
			if empirical_lookup_value in self.predictors:
				index = self.predictors.index(empirical_lookup_value)

				if empirical_lookup_value == 'emp_product_look_id':
					if X_test[column] in self.product_lookid_lookup_table.index:
						listofzeros[index] = float(self.product_lookid_lookup_table['emp_product_look_id'][self.product_lookid_lookup_table.index == X_test[column]])
					else:
						listofzeros[index] = self.mean_predictor_values[empirical_lookup_value]


				if empirical_lookup_value == 'emp_sale_id':
					if X_test[column] in self.product_lookid_lookup_table.index:
						listofzeros[index] = float(self.sale_id_lookup_table['emp_sale_id'][self.sale_id_lookup_table.index == X_test[column]])
					else:
						listofzeros[index] = self.mean_predictor_values[empirical_lookup_value]

		
			if column in self.predictors:

				index = self.predictors.index(column)

				# Checks it is a numerical type, if not prints a message and replaces with average
				# This can be used in cases when a particular value is not available like NaN
				if isinstance(X_test[column], float) or isinstance(X_test[column], int):
					listofzeros[index] = float(X_test[column])
				else:
					# If value is not valid or missing replace with mean
					listofzeros[index] = self.mean_predictor_values[column]

		X_sample = pd.Series(listofzeros, index = self.predictors)

		return X_sample


	def online_predict_value(self, X_test, columns):

		X_sample = self.derive_predictors(X_test, columns)

		return self.trained_model.predict(X_sample.values.reshape(1, -1))[0]

	def batch_predict_value(self, path):

		sales_df = pd.read_csv(path, delimiter=',')

		predictions = []
		for index, row in sales_df.iterrows():
			predictions.append(self.online_predict_value(row, sales_df.columns))

		return predictions

	def load_test(self, path):

		sales_df = pd.read_csv(path, delimiter=',')

		predictions = []
		lagged_time = 0
		count = 0
		for index, row in sales_df.iterrows():

			start_time = time.time()
			predictions.append(self.online_predict_value(row, sales_df.columns))
			end_time = time.time()

			lagged_time = lagged_time + end_time - start_time

			count = count + 1
			if count > 100:
				break

		return lagged_time / count




micro_service = PredictionMicroService()

micro_service.model_total_sales(0.0001)

# sales rate expressed as number of sales per day
# micro_service.model_sales_rate(0.0001)

micro_service.save_model(".")
micro_service.load_model(".")

sales_df = pd.read_csv("Gilt_datascience_exercise.csv", delimiter=',')

# Just use the first record as a dummy sample
prediction = micro_service.online_predict_value(sales_df.iloc[0], sales_df.columns)

print "##################### Prediction #####################"

print "prediction", prediction

avg_latency = batch_predictions = micro_service.load_test("Gilt_datascience_exercise.csv")

print "Avg Latency (s) ", avg_latency


