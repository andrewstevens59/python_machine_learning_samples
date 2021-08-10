

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
from maximize_sharpe import *

import delta_process as delta_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes

import numpy as np
import pandas as pd 


def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 20
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt).tolist(), returns, risks

def align_results_series(results, times, currency_pairs):

    global_times = []
    for pair in range(len(currency_pairs)):
        global_times += times[pair]

    global_times = list(set(global_times))

    time_map = {}
    for time in global_times:
        time_map[time] = len(time_map)

    compacted_returns = np.zeros((len(global_times), len(currency_pairs)))
    for index in range(len(currency_pairs)):

        for result, time in zip(results[index], times[index]):
            compacted_returns[time_map[time]][index] = result

    return compacted_returns



def calculate_portfolio(results, currency_pairs, times):

    equity = []
    global_returns = align_results_series(results, times, currency_pairs)
    copy_returns = copy.deepcopy(global_returns)

    wt = max_sharpe_ratio(global_returns, optimization_bounds = (0.0, 1.0))


    offset = 0
    curr_return = 0
    final_returns = []
    for index in range(0, len(global_returns)):

        
        net_return = 0
        for pair_id in range(len(currency_pairs)):
            curr_return += copy_returns[index][pair_id] * wt[pair_id]
            net_return += copy_returns[index][pair_id] * wt[pair_id]

        final_returns.append(net_return)

        equity.append(curr_return)
        offset += 1

    sharpe = (np.mean(final_returns) / np.std(final_returns)) * math.sqrt(24 * 252)
    #print "Est Sharpe Ratio", sharpe

    #print "Weights", wt


    return wt, currency_pairs, sharpe


def min_exposure_weights(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)

    return p_var

def find_min_exposure_weights(returns, optimization_bounds = (-1.0, 1.0)):

    df = pd.DataFrame(returns)
    mean_returns = df.mean()
    cov_matrix = df.cov()

    num_assets = len(mean_returns)
    avg_weight = 1.0 / num_assets

    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
    args = (mean_returns, cov_matrix, 0)
    bound = optimization_bounds
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(min_exposure_weights, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    weights = result['x'].tolist() 
    return weights

def minimize_exposure(results, currency_pairs, times):


    global_exposures = align_results_series(results, times, currency_pairs)

    wt = find_min_exposure_weights(global_exposures, optimization_bounds = (0.0, 1.0))






