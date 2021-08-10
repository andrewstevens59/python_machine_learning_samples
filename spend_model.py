


df = pd.DataFrame()

activity = "online_activity"
spend = "online_spend"


crns = df['crn'].values.tolist()

def create_feature_set(last_day_index):

	X = []
	y = []
	for crn in crns:

		sub_df = df[(df[online_spend] == 1) && (df['crn'] == crn)]

		min_date = df['date'].min()
		max_date = df['date'].max()

		dates = sub_df['date'].values.tolist()

		lag_dist = [0] * 32

		avg_spends = []

		curr_date = dates[-last_day_index - 1]

		count = 0
		for index in range(len(dates)-last_day_index):

			lag = (dates[index+1] - dates[index]).days

			lag_dist[lag] += 1
			count += 1

		for day in range(60):

			before_date = curr_date - time_delta(days=day)

			count = len(sub_df[(sub_df['date'] >= before_date) & (sub_df['date'] <= curr_date)])

			if(spend_count > 0):
				avg_spends.append(sub_df[spend][(sub_df['date'] >= before_date) & (sub_df['date'] <= curr_date)].sum() / count)
			else:
				avg_spends.append(0)


		lag_dist = [float(v) / count for v in lag_dist] + avg_spends

		X.append(lag_dist)
		y.append(sub_df['spend'][sub_df['date'] == dates[-last_day_index]])

	return X, y

X_train, y_train = create_feature_set(2)
X_test, y_test = create_feature_set(1)







