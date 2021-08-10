

import sys
import math
from datetime import datetime
from random import *
import os.path


import pickle

import pycurl
from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA

import time
import datetime
import calendar
import json
import copy

import pickle
import math
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs


from sklearn.linear_model import LinearRegression

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import delta_process as delta_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import scipy.optimize as sco

import pandas as pd  
import numpy as np

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) 
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) 
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)

    net_weight = 0 
    for w, r in zip(weights, mean_returns):
        net_weight += 4 * w * w

    return -(1 / p_var) + net_weight

def max_sharpe_ratio(returns, optimization_bounds = (-1.0, 1.0)):

    df = pd.DataFrame(returns)
    mean_returns = df.mean()
    cov_matrix = df.cov()

    num_assets = len(mean_returns)
    avg_weight = 1.0 / num_assets

   # constraints = [{'type': 'ineq', 'fun': lambda x: +x[i] - avg_weight * max_exposure} for i in range(num_assets)]
    
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
    args = (mean_returns, cov_matrix, 0)
    bound = optimization_bounds
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    weights = result['x'].tolist() 
    return weights
  
def random_portfolios(num_portfolios, returns, max_exposure = 2.2):

    np.random.seed(0)

    df = pd.DataFrame(returns)
    mean_returns = df.mean()
    cov_matrix = df.cov()
    avg_weight = 1.0 / len(mean_returns)

    results = []
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))

        for model_index in range(len(mean_returns)):
            if mean_returns[model_index] < 0:
                weights[model_index] = 0

        if np.sum(weights) == 0:
            results = max_sharpe_ratio(returns)
            weights = results['x']
            portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)

            return portfolio_return / portfolio_std_dev, weights.tolist()


        weights /= np.sum(weights)
        

        is_invalid = False

        if max_exposure != None:
            for w in weights:
                if abs(w) > avg_weight * max_exposure:
                    is_invalid = True
                    break

        if is_invalid == True:
            continue

        weights_record.append(weights)
        
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)

        results.append([portfolio_std_dev, portfolio_return, portfolio_return / portfolio_std_dev, len(results)])

    sorted_set = sorted(results, key=lambda x: abs(x[2]), reverse=True)

    sorted_set = sorted_set[:10]
    sorted_set = sorted(results, key=lambda x: abs(x[1]), reverse=True)

    max_index = sorted_set[0][3]

    return sorted_set[0][2], weights_record[max_index].tolist()
