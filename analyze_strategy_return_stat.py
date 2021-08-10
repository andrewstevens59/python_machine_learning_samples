import ast
import json
import time
import sys
import glob
import calendar
import datetime
from dateutil.tz import *
import numpy as np
import scipy.optimize as sco
import pandas as pd
import os
import pickle


files1 = glob.glob("/root/trade_news_release*.log")
files2 = glob.glob("/root/stat_arb_trading*.log")
files = files1 + files2
to_zone = tzutc()

all_currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


def tail( f, lines=20 ):
    total_lines_wanted = lines

    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = [] # blocks of size BLOCK_SIZE, in reverse order starting
                # from the end of the file
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            # read the last block we haven't yet read
            f.seek(block_number*BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count('\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = ''.join(reversed(blocks))
    return all_read_text.splitlines()[-total_lines_wanted:]

def calculate_sharpe(file_name):

    return 0

    with open(file_name, "r") as fi:
        text_rows = tail(fi, 75000)
        
    equity_vals = []
    prev_time = 0
    for line in text_rows:
      
        try:
            curr_time = calendar.timegm(datetime.datetime.strptime(line[:len("2018-09-19 10:08:08")], "%Y-%m-%d %H:%M:%S").timetuple()) 

            ln = line[len("2018-09-21 12:23:05 "):]

            if ln.startswith("Equity:") and abs(curr_time - prev_time) > 24 * 60 * 60:
                equity_vals.append(float(ast.literal_eval(ln[len("Equity: "):])))
                prev_time = curr_time
        except:
            pass


    inc_returns = [v1 - v2 for v1, v2 in zip(equity_vals[1:], equity_vals[:-1])]
    return np.mean(inc_returns) / np.std(inc_returns)

def update_weekly_returns(file_name, portfolio_return_map):

    with open(file_name, "r") as fi:
        text_rows = tail(fi, 3000)

    for line in text_rows:

        try:

            ln = line[len("2018-09-21 12:23:05 "):]

            key = datetime.datetime.strptime(line[:len("2018-09-19 10:08:08")], "%Y-%m-%d %H:%M:%S").strftime("%Y %V")
       
            if ln.startswith("Equity:"):
                equity = float(ast.literal_eval(ln[len("Equity: "):]))

                if key not in portfolio_return_map:
                    portfolio_return_map[key] = {}

                portfolio_return_map[key][file_name] = equity

        except:
            pass

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) 
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) 
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_std, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)

    return -(p_ret / p_std)

def max_sharpe_ratio(columns, returns, optimization_bounds = (-1.0, 1.0)):

    df = pd.DataFrame(returns, columns = columns)

    mean_returns = df.mean()
    std_returns = df.std()
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
    return weights, [a / b for a, b in zip(mean_returns, std_returns)]


root_dir = "/root/"

def find_portfolio_weights(portfolio_return_map):

    dates = sorted(portfolio_return_map.keys())[-24:]

    column_map = {}
    for date in portfolio_return_map:
        for setting_name in portfolio_return_map[date]:
            if "normalize_signal_1" in setting_name:
                continue

            if setting_name not in column_map:
                column_map[setting_name] = len(column_map)

    print (dates)

    returns = []
    for index in range(len(dates)-1):

        curr_date = dates[index]
        next_date = dates[index+1]

        row = [0] * len(column_map)
        for setting_name in column_map:

            i = column_map[setting_name]

            if setting_name == "/root/params_rmse_percentile_95th_percentile_min_volatility_125_is_normalize_signal_0_trade_news_release_no_hedge_all.log":
                print (row[i], "$$$",  setting_name in portfolio_return_map[curr_date], setting_name in portfolio_return_map[next_date])

            if setting_name in portfolio_return_map[next_date] and setting_name in portfolio_return_map[curr_date]:
                row[i] = (portfolio_return_map[next_date][setting_name] - portfolio_return_map[curr_date][setting_name]) / portfolio_return_map[curr_date][setting_name]

                if setting_name == "/root/params_rmse_percentile_95th_percentile_min_volatility_125_is_normalize_signal_0_trade_news_release_no_hedge_all.log":
                    print (row[i], "&&&", curr_date, next_date)
        

        returns.append(row)

    columns = [[column_map[setting_name], setting_name] for setting_name in column_map]
    columns = sorted(columns, key=lambda x: x[0])
    columns = [column[1] for column in columns]

    print (returns)
    weights, sharpes = max_sharpe_ratio(columns, returns, optimization_bounds = (0.0, 1.0))
    
    for w in [weights, sharpes]:
        weights = ([[a, b] for a, b in zip(w, columns) if a > 0.001])

        final_map = {}
        for port_weight in weights:
            final_map[port_weight[1]] = port_weight[0]

        print (final_map)

        if w == weights:
            with open(root_dir + "portfolio_weights.pickle", "wb") as f:
                pickle.dump(final_map, f)
        else:
            with open(root_dir + "portfolio_sharpes.pickle", "wb") as f:
                pickle.dump(final_map, f)



if os.path.isfile(root_dir + "strategy_return_stat"):
    with open(root_dir + "strategy_return_stat", "rb") as f:
        group_metadata = pickle.load(f)
else:
    group_metadata = {}


returns = []
return_params = []
return_map = {"all_no_hedge" : [], "all" : [], "no_hedge" : [], "" : [], "portfolio" : [], "portfolio_all" : []}
for file_name in files:
    print (file_name)
    if "stat_arb_trading" not in file_name:
        continue

    if "low_barrier" in file_name:
        continue

    for pair in all_currency_pairs:
        if pair in file_name:
            select_pair = pair
            break


    equity_vals = []
    with open(file_name, "r") as fi:
        for line in fi:

            try:

                ln = line[len("2018-09-21 12:23:05 "):]

                if ln.startswith("Equity:"):
                    equity_vals.append(float(ast.literal_eval(ln[len("Equity: "):])))
                    break
            except:
                pass

    with open(file_name, "r") as fi:
        text_rows = tail(fi, 300)

    for line in text_rows:

        try:

            ln = line[len("2018-09-21 12:23:05 "):]

            if ln.startswith("Equity:"):
                equity_vals.append(float(ast.literal_eval(ln[len("Equity: "):])))
        except:
            pass

    if len(equity_vals) < 2:
        continue

    ret = (equity_vals[-1] - equity_vals[0]) * (1.0 / equity_vals[0])

    
    print (file_name, ret, calculate_sharpe(file_name))
    returns.append(ret)

if len(sys.argv) >= 2:
    find_portfolio_weights(group_metadata)
    with open(root_dir + "strategy_return_stat", "wb") as f:
        pickle.dump(group_metadata, f)

print ("Total Return", sum(returns))


