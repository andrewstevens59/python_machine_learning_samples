import execute_news_signals
import mysql.connector
import numpy as np
import pickle
import os

all_currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

execute_news_signals.checkIfProcessRunning("execute_managed_account_portfolios.py", "")



cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

cursor = cnx.cursor()
query = ("SELECT t1.*, t2.is_hedged FROM managed_strategies t1, managed_accounts t2 where t1.user_id=t2.user_id and t1.account_nbr=t2.account_nbr")

cursor.execute(query)

setup_rows = []
for row1 in cursor:
    setup_rows.append(row1)

cursor.close()


cursor = cnx.cursor()
query = ("SELECT count(*) as count, user_id, account_nbr FROM managed_strategies group by user_id, account_nbr")

cursor.execute(query)

account_strategy_num = {}
for row1 in cursor:
    account_strategy_num[str(row1[1]) + "_" + str(row1[2])] = row1[0]

cursor.close()

return_map = pickle.load(open("/root/portfolio_pairs.pickle", "rb"))

row_count = 0
for row in setup_rows:

    user_id = row[0]
    account_nbr = row[1]
    execute_news_signals.api_key = row[2]
    select_pair = row[3]
    strategy = row[4]
    is_demo = row[5]
    is_max_barrier = row[6]
    strategy_weight = row[7]
    is_hedge = row[8]

    if strategy != "P1" and strategy != "P2":
        continue

    print row

    if strategy == "P2":
        if is_hedge == False:
            key = "all_no_hedge"
        else:
            key = "all"
    else:
        if is_hedge == False:
            key = "no_hedge"
        else:
            key = ""

    trade_pairs = [item["pair"] for item in return_map[key]]
    sharpes = [item["sharpe"] for item in return_map[key]]
    mean_sharpe = np.mean(sharpes)

    strategy_count = account_strategy_num[str(user_id) + "_" + str(account_nbr)]


    print (user_id)

    trade_pairs = [item["pair"] for item in return_map[key]] 

    print ("Mean Wt", mean_sharpe, trade_pairs)
 
    execute_news_signals.file_ext_key = "_" + str(user_id) + "_" + account_nbr
    execute_news_signals.root_dir = "/root/user_data/" + str(user_id) + "/" + account_nbr + "/"

    if os.path.isdir(execute_news_signals.root_dir) == False:
        os.makedirs(execute_news_signals.root_dir)

    if row_count > 0:
        handlers = execute_news_signals.trade_logger.handlers[:]
        for handler in handlers:
            handler.close()
            execute_news_signals.trade_logger.removeHandler(handler)

    execute_news_signals.trade_logger = execute_news_signals.setup_logger('first_logger', execute_news_signals.root_dir + "trade_managed_accounts.log")

    if is_demo:
        execute_news_signals.account_type = "fxpractice"
    else:
        execute_news_signals.account_type = "fxtrade"

    for select_pair in all_currency_pairs:

        if strategy == "P1":
            #sharpe = execute_news_signals.get_sharpe(False, is_hedge, select_pair, "/root/news_signal_", execute_news_signals.ModelType.barrier) 

            print ("strategy_weight", strategy_weight)

            execute_news_signals.process_pending_trades([account_nbr], 
                execute_news_signals.avg_spreads, select_pair, "/root/news_signal_", 
                execute_news_signals.ModelType.barrier, is_low_barrier = False, 
                strategy_weight = strategy_weight, is_new_trade = (select_pair in trade_pairs)) 
        elif strategy == "P2":
            #sharpe = execute_news_signals.get_sharpe(False, is_hedge, select_pair, "/root/news_signal_all_", execute_news_signals.ModelType.barrier) 

            print ("strategy_weight", strategy_weight)

            execute_news_signals.process_pending_trades([account_nbr], 
                execute_news_signals.avg_spreads, select_pair, "/root/news_signal_all_", 
                execute_news_signals.ModelType.barrier, is_low_barrier = False, 
                strategy_weight = strategy_weight, is_new_trade = (select_pair in trade_pairs)) 

    row_count += 1


