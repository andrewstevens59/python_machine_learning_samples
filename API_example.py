from quantforextrading import api

data_api = api.HistoricalDataAPI()

rates_df = data_api.get_historical_rates(currency_pair = "EUR_USD", year = "2015", columns = ["bid", "ask", "timestamp"])

for index, rates in rates_df.iterrows():

	price = (rates["bid"] + rates["ask"]) / 2

	timestamp = rates["timestamp"]

	prediction_df = data_api.get_historical_predictions(currency_pair = "EUR_USD", timestamp = timestamp, timeframe_units = "H", timeframe_records = 24)


	barrier = data_api.calculate_expected_barrier(predictions = prediction_df, auc_cutoff = "0.6", probability_cutoff = "0.9", signals = ["NewsModelA", "NewsModelB"])

	if barrier > 100:


data_api = api.HistoricalDataAPI()

rates_df = data_api.get_live_rates(currency_pair = "EUR_USD", timeframe_units = "H", timeframe_records = 24, columns = ["bid", "ask", "timestamp"])

for index, rates in rates_df.iterrows():

	price = (rates["bid"] + rates["ask"]) / 2

	timestamp = rates["timestamp"]

	prediction_df = data_api.get_live_predictions(currency_pair = "EUR_USD", timestamp = timestamp, timeframe_units = "H", timeframe_records = 24)


	barrier = data_api.calculate_expected_barrier(predictions = prediction_df, auc_cutoff = "0.6", probability_cutoff = "0.9", signals = ["NewsModelA", "NewsModelB"])

from quantforextrading import api
data_api = api.HistoricalDataAPI()

class Order:

    def __init__(self):
        self.open_price = 0
        self.time = 0
        self.amount = 0

data_api = api.HistoricalDataAPI()

rates_df = data_api.get_historical_rates(currency_pair = "EUR_USD", year = "2015", columns = ["bid", "ask", "timestamp"])


orders = []
curr_trade_dir = None
total_profit = 0
equity = []


for index, rates in rates_df.iterrows():
	exchange_rate = (rates["bid"] + rates["ask"]) / 2

	timestamp = rates["timestamp"]

	prediction_df = data_api.get_historical_predictions(currency_pair = "EUR_USD", timestamp = timestamp, timeframe_units = "H", timeframe_records = 24)

	barrier = data_api.calculate_expected_barrier(predictions = prediction_df, auc_cutoff = "0.6", probability_cutoff = "0.9", signals = ["NewsModelA", "NewsModelB"])

	pnl = 0
	for order in orders:
		if order.dir == (exchange_rate > order.open_price):
			pnl += abs(order.open_price - exchange_rate) * order.amount
		else:
			pnl -= abs(order.open_price - exchange_rate) * order.amount

	equity.append(total_profit + pnl)

	if abs(barrier) > 100:

		if pnl > 0:
			total_profit += pnl
			curr_trade_dir = None
			orders = []

		if ((signal > 0) != curr_trade_dir):
			order = Order()
			order.open_price = exchange_rate
			order.trade_dir = barrier > 0
			order.amount = abs(signal) * (0.0001 + abs(pnl))
			curr_trade_dir = order.trade_dir
			orders.append(order)
