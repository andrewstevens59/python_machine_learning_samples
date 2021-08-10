class UsagePrediction():
	
	def __init__(self):
		self.df1 = pd.read_csv('dataset_1.csv')
		self.df2 = pd.read_csv('dataset_2.csv')
		self.clf = GradientBoostingRegressor()
		
		self.df1.fillna(method='ffill', inplace=True)
		self.df2.fillna(method='ffill', inplace=True)
		
		self.X_columns = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3',
	   'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
	   'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
	   'Visibility', 'Tdewpoint', 'v1', 'v2']

	
	def create_feature_set1(self, df):
		return df[self.X_columns]
	
	
	def create_feature_set2(self, df):
		X = []
		for i in range(len(df)):
			
			features = []
			for lag in range(7):
				features += df.iloc[i-lag:i].mean()
				
			X.append(features)
			
		return self.df1[self.X_columns]
	
	def train_model(self):
		self.clf.fit(np.array(self.df1[self.X_columns]), self.df1['Appliances'])
		
	def evaluate_model(self):

		X = self.df1[self.X_columns]
		y = self.df1['Appliances']
		
		r2_scores = []
		for it_num in range(10):
			print (it_num)
			X_train, X_test, y_train, y_test = train_test_split(
			 X, y, test_size=0.33, random_state=42)

			clf = GradientBoostingRegressor()
			clf.fit(X_train, y_train)

			predictions = clf.predict(X_test)

			r2 = r2_score(y_test, predictions)
			r2_scores.append(r2)
			
		return np.mean(r2_scores)
	
	def make_predictions(self):
		return self.clf.predict(np.array(self.df2[self.X_columns])
		
model = UsagePrediction()
model.train_model()
print (model.evaluate_model())
		
		
		
		
		
	
