
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

'''
	Description: This is a mockup for a ctr model that is intended for my project
	www.demandmaretapp.com. The project has a swiper component which requires
	predictions regarding which items to show a user based on their past swipe history.
	The features used in the CTR model are derived from topic distributions and word vectors.

	Features: The features for the CTR model consisted of
		1. A topic distribution for each item
		2. A topic distribution for each user
		3. A word vector for each user_id
		4. A word vector for each item_id

	The final word vector provided as a feature is the difference between the 
	user word vector and the item word vector. The topic distribution is used 
	to help group users and items that are similar based on clustering of 
	swipes. The final RL model basically combines multiple metrics regarding
	the similarity of a user and an item to predict a swipe.

	Output: The output ranks possible items provided by the probabity of a swipe.

'''

def cross_val_calculator(X, y, cross_val_num, is_sample_wt, params = None):

	y_true_indexes = [index for index in range(len(y)) if y[index] == True]
	y_false_indexes = [index for index in range(len(y)) if y[index] == False]

	y_test_all = []
	y_preds_all = []
	for iteration in range(cross_val_num):

		rand.seed(iteration)
		rand.shuffle(y_true_indexes)

		rand.seed(iteration)
		rand.shuffle(y_false_indexes)

		min_size = max(15, int(min(len(y_false_indexes), len(y_true_indexes)) * 0.35))
		if min_size >= max(len(y_true_indexes), len(y_false_indexes)) * 0.8:
			return -1

		true_indexes = y_true_indexes[:min_size]
		false_indexes = y_false_indexes[:min_size]

		X_train = []
		y_train = []

		X_test = []
		y_test = []
		for index in range(len(y)):
			if index in true_indexes + false_indexes:
				X_test.append(X[index])
				y_test.append(y[index])
			else:
				X_train.append(X[index])
				y_train.append(y[index])

		if params == None:
			clf = xgb.XGBClassifier()
		else:
			clf = xgb.XGBClassifier(
		        max_depth=int(round(params["max_depth"])),
		        learning_rate=float(params["learning_rate"]),
		        n_estimators=int(params["n_estimators"]),
		        gamma=params["gamma"])

		if is_sample_wt:
			
			true_wt = float(sum(y_train)) / len(y_train)
			false_wt = 1 - true_wt

			weights = []
			for y_s in y_train:
				if y_s:
					weights.append(false_wt)
				else:
					weights.append(true_wt)

			clf.fit(np.array(X_train), y_train, sample_weight=np.array(weights))
		else:
			clf.fit(np.array(X_train), y_train)

		preds = clf.predict_proba(np.array(X_test))[:,1]

		y_test_all += y_test
		y_preds_all += list(preds)

	fpr, tpr, thresholds = metrics.roc_curve(y_test_all, y_preds_all)

	return metrics.auc(fpr, tpr)

def bayesian_optimization_output(X_train, y_train):

    
    pbounds = {
	    'learning_rate': (0.01, 1.0),
	    'n_estimators': (100, 1000),
	    'max_depth': (3,10),
	    'gamma': (0, 5)}

	
	def xgboost_hyper_param(learning_rate,
                        n_estimators,
                        max_depth,
                        gamma):
 
	    max_depth = int(max_depth)
	    n_estimators = int(n_estimators)
	 
	    clf = xgb.XGBClassifier(
	        max_depth=max_depth,
	        learning_rate=learning_rate,
	        n_estimators=n_estimators,
	        gamma=gamma)

        return cross_val_calculator(X_train, y_train, 10, False, params = None):
     
     
    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=8,
        n_iter=16,
    )

    xgb.XGBClassifier(
	        max_depth=optimizer.max['params']['max_depth'],
	        learning_rate=optimizer.max['params']['learning_rate'],
	        n_estimators=optimizer.max['params']['n_estimators'],
	        gamma=optimizer.max['params']['gamma'])

    return optimizer.max['params'], optimizer.max['target']

def get_user_items(user_ids, item_ids):

	user_id_items = {}

	for user_id, item_id in zip(user_ids, item_ids):
		if user_id not in user_id_items:
			user_id_items[user_id] = []

		user_id_items[user_id].append(item_id)

	return user_id_items

def get_item_users(user_ids, item_ids):

	user_id_items = {}

	for user_id, item_id in zip(user_ids, item_ids):
		if item_id not in user_id_items:
			user_id_items[item_id] = []

		user_id_items[item_id].append(user_id)

	return user_id_items

def find_topics_for_users(user_ids, item_ids):

	user_id_items = get_user_items(user_ids, item_ids)
	ldamodel = gensim.models.ldamodel.LdaModel(user_id_items.values(), num_topics=50, alpha='auto', eval_every=5)
	ldamodel.save('model5.gensim')

	topic_feature_map = {}
	for user_id in user_id_items:
		topic_feature_map[user_id] = ldamodel[user_id_items[user_id]]

 	return topic_feature_map, ldamodel

 def find_topics_for_items(user_ids, item_ids):

	user_id_items = get_item_users(user_ids, item_ids)
	ldamodel = gensim.models.ldamodel.LdaModel(user_id_items.values(), num_topics=50, alpha='auto', eval_every=5)
	ldamodel.save('model5.gensim')

	topic_feature_map = {}
	for item_id in user_id_items:
		topic_feature_map[item_id] = ldamodel[user_id_items[user_id]]

 	return topic_feature_map, ldamodel

def find_user_vectors(user_ids, item_ids):

	user_id_items = get_user_items(user_ids, item_ids)
	model = Word2Vec(user_id_items.values(), size=20, window=5, min_count=1, workers=4)

	model.train(user_id_items.values(), total_examples=1, epochs=1)

	user_word_vectors = {}
	for user_id in user_id_items:

		net_word_vector = [0] * 20
		for item_id in user_id_items[user_id]:
			net_word_vector = [a + b for a, b in zip(model.wv[item_id], net_word_vector)]

		user_word_vectors[user_id] = net_word_vector

	return user_word_vectors, model.wv

def train_recommendation(user_ids, swipe_labels, item_ids):

	topic_user_map, ldamodel = find_topics_for_users(user_ids, item_ids)
	topic_item_map, ldamodel = find_topics_for_items(user_ids, item_ids)

	user_word_vectors, wv_map = find_user_vectors(user_ids, item_ids)

	X_train = []
	for user_id, item_id in zip(user_ids, item_ids):
		topic_user_vector = topic_user_map[user_id]
		topic_item_vector = topic_item_map[item_id]

		user_word_vector = user_word_vectors[user_id]
		item_vector = wv_map[item_id]
		wv_feature = [a - b for a, b in zip(item_vector, user_word_vector)]

		X_train.append(topic_user_vector + topic_item_vector + wv_feature)

	clf = bayesian_optimization_output(X_train, swipe_labels)

	with open("ctr_model.pickle", "wb") as f:
		pickle.dump(clf, f)

	with open("wv_map.pickle", "wb") as f:
		pickle.dump(wv_map, f)

	with open("ldamodel.pickle", "wb") as f:
		pickle.dump(ldamodel, f)

	with open("user_word_vectors.pickle", "wb") as f:
		pickle.dump(user_word_vectors, f)

def rank_most_likely(user_id, possible_item_ids, clf, wv_map, topic_user_map, topic_item_map, user_word_vectors):

	user_word_vector = user_word_vectors[user_id]

	probs = []
	for item_id in possible_item_ids:
		item_vector = wv_map[item_id]

		wv_feature = [a - b for a, b in zip(item_vector, user_word_vector)]
		prob = clf.predict_proba([topic_user_map[user_id] + topic_item_map[item_id] + wv_feature])[0][1]

		probs.append([prob, item_id])

	return sorted(probs, lambda x: x[0], reverse = True)




