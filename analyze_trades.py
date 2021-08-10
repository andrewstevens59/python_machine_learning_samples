
import ast
import json

print "scalper"
with open("/root/trade_news_trend.log","r") as fi:
    lines = []
    for ln in fi:
    	ln = ln[len("2018-09-21 12:23:05 "):]
        if ln.startswith("Close Model Order:"):
            lines.append(ast.literal_eval(ln[len("Close Model Order: "):]))

total_profit = 0
profit_by_model = {}
trades_by_model = {}

equity_curve = 0
curr_equity = {}
max_equity = {}
for ln in lines:

	total_profit += float(ln['Actual Profit'])
	model_key = ln['model_key']

	model_key = model_key[:len('EUR_GBP')]


	if model_key not in curr_equity:
		curr_equity[model_key] = 0
		max_equity[model_key] = 0


	max_equity[model_key] = max(curr_equity[model_key], max_equity[model_key])

	if model_key not in profit_by_model:
		profit_by_model[model_key] = 0
		trades_by_model[model_key] = 0

	profit_by_model[model_key] += float(ln['Actual Profit'])
	trades_by_model[model_key] += 1


sorted_set = []
for key in profit_by_model:
	sorted_set.append([key, profit_by_model[key], trades_by_model[key]])

sorted_set = sorted(sorted_set, key=lambda x: (x[1]), reverse=True)

for item in sorted_set:
	print item[0], item[1], item[2]

print "Total Profit", total_profit






with open("/root/trade_news_trend.log","r") as fi:
    lines = []
    for ln in fi:
    	ln = ln[len("2018-09-21 12:23:05 "):]
        if ln.startswith("Close Not Exist Order:"):
            lines.append(ast.literal_eval(ln[len("Close Not Exist Order: "):]))

total_profit = 0
for ln in lines:

	try:
		total_profit += float(ln["orderFillTransaction"]["pl"])
	except:
		pass

print "Error Closed", total_profit

