from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def predict(train_start, train_end, x, labels, is_use_residual):


	x_train = x[train_start:train_end]
	y_train = labels[train_start:train_end]

	mean = np.mean(y_train)
	y_train = [(v - mean) for v in y_train]

	clf = GradientBoostingRegressor(random_state=42)
	clf.fit(x_train[:-200], y_train[:-200])

	if is_use_residual == False:
		biased_predictions = clf.predict(x_train)

		mean = np.mean(biased_predictions)
		std = np.std(biased_predictions)

		return (biased_predictions[-1] - mean) / std

	biased_predictions = clf.predict(x_train[-200:])
	residuals = [v - p for v, p in zip(biased_predictions, y_train[-200:])]

	mean = np.mean(residuals)
	std = np.std(residuals)

	return (residuals[-1] - mean) / std