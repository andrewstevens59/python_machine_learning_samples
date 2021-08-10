import execute_news_signals
import mysql.connector
import numpy as np
import os

execute_news_signals.checkIfProcessRunning("execute_managed_account_signals.py", "")



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

    if strategy != "S1" and strategy != "S2":
        continue

    print row

    strategy_count = account_strategy_num[str(user_id) + "_" + str(account_nbr)]

    '''
    if account_nbr == "001-011-2949857-005":
        strategy_weight = 10
    '''

    print ("Mean Wt", strategy_weight)

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


    if strategy == "S1":
        execute_news_signals.process_pending_trades([account_nbr], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_", execute_news_signals.ModelType.barrier, is_low_barrier = False, strategy_weight = strategy_weight) 
    elif strategy == "S2":
        execute_news_signals.process_pending_trades([account_nbr], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_all_", execute_news_signals.ModelType.barrier, is_low_barrier = False, strategy_weight = strategy_weight) 
    elif strategy == "S3":
        execute_news_signals.process_pending_trades([account_nbr], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_", execute_news_signals.ModelType.barrier, is_low_barrier = True, strategy_weight = strategy_weight) 
    elif strategy == "S4":
        execute_news_signals.process_pending_trades([account_nbr], execute_news_signals.avg_spreads, select_pair, "/root/news_signal_all_", execute_news_signals.ModelType.barrier, is_low_barrier = True, strategy_weight = strategy_weight) 

    row_count += 1


