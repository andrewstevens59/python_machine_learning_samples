import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from random import *
import time
import os.path

from sklearn import datasets, linear_model


from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import numpy as np
from sklearn.decomposition import PCA

import pickle

sys.path.append('download_calendar.py')
sys.path.append('download_rates.py')

from download_rates import download_rates
from download_calendar import download_calendar

training_days = 15
calendar_backdays = 20
currencies = ['USD', 'EUR', 'NZD', 'AUD', 'CAD', 'GBP', 'JPY', 'CHF']
currency_num = len(currencies)
impact_num = 4

total_observed_days = 8

curr_time = int(time.time())

singular_feature_num = training_days * currency_num * impact_num

chosen_currency_pair = "EUR_USD"

save_model_dir = "/Users/callum/Documents/" + chosen_currency_pair + "_"

def featureId(day, impact, currency):
    return day + (training_days * impact) + (currency * training_days * impact_num)


def skipBack(time):
    day = datetime.fromtimestamp(time).strftime("%A")

    if day == 'Saturday':
        return time - (60 * 60 * 24 * 2)

    if day == 'Sunday':
        return time - (60 * 60 * 24)

    return time

def skipForward(time):
    day = datetime.fromtimestamp(time).strftime("%A")

    if day == 'Saturday':
        return time + (60 * 60 * 24 * 2)

    if day == 'Sunday':
        return time + (60 * 60 * 24)

    return time

def findCalendarFeatures(start, end, calendar, feature_vector):
    snapshot = calendar[(calendar.index >= start) & (calendar.index < end)]


    for currency in range(currency_num): 
        usd = snapshot[snapshot.currency == currencies[currency]]
        if len(usd) == 0:
            continue

        for index, row in usd.iterrows():

            day = int((float(index - start) / float(end - start)) * training_days)

            try:
                if float(row['forecast']) == 0:
                    continue
            except ValueError:
                continue

            impact = row['impact']
            feature_id = featureId(day, impact, currency)
            feature_vector[feature_id] = feature_vector[feature_id] + ((float(row['actual']) - float(row['forecast'])) / float(row['forecast']))

    return feature_vector

def calcPercChange(start_time, end_time, rates):
    

    price = rates[(rates.index >= start_time) & (rates.index <= end_time)]['rate']

    if len(price) == 0:
        return 0

    end_price = float(price.iloc[len(price)-1])

    perc_change = (end_price - price.iloc[0]) 

    return perc_change

def lookupPCAFeatures(input_set, feature_offset, observed_days):

    pca = pickle.load(open(save_model_dir + "pca_" + str(observed_days) + "_" + str(feature_offset), 'rb'))
    bins = pickle.load(open(save_model_dir + "quantiles_" + str(observed_days) + "_" + str(feature_offset), 'rb'))
    lookup_table = pickle.load(open(save_model_dir + "lookup_table_" + str(observed_days) + "_" + str(feature_offset), 'rb'))

    bins1 = bins[0]
    bins2 = bins[1]

    X = pca.transform(input_set)

    pca1 = [float(row[0]) for row in X]
    pca2 = [float(row[1]) for row in X]

    pca1 = np.digitize(pca1, bins1)
    pca2 = np.digitize(pca2, bins2)
    X = zip(pca1, pca2)

    pca_features = []

    for (a, b) in X:

        grid_pos = min(9, int(a)) + (min(9, int(b)) * 10)

        if grid_pos in lookup_table:
            pca_features.append(lookup_table[grid_pos])
        else:
            pca_features.append(lookup_table[-1])

    return pca_features
        

def createPCAFeatures(input_set, input_labels, feature_offset, observed_days, grid_size=5):


    pca_dim = 2
    pca = PCA(n_components=pca_dim)
    X = pca.fit_transform(input_set)

    pickle.dump(pca, open(save_model_dir + "pca_" + str(observed_days) + "_" + str(feature_offset), 'wb'))

    pca1 = [float(row[0]) for row in X]
    pca2 = [float(row[1]) for row in X]
      
    bins1 = np.percentile(pca1, [float(0.2) * i for i in range(6)]).tolist()
    bins2 = np.percentile(pca2, [float(0.2) * i for i in range(6)]).tolist()
 
    bins1 = [bins1[i] + i * 0.000000000001 for i in range(len(bins1))] 
    bins2 = [bins2[i] + i * 0.000000000001 for i in range(len(bins2))]

    pickle.dump([bins1, bins2], open(save_model_dir + "quantiles_" + str(observed_days) + "_" + str(feature_offset), 'wb'))

    pca1 = np.digitize(pca1, bins1)
    pca2 = np.digitize(pca2, bins2)
    X = zip(pca1, pca2)


    import bisect 

    label_offset = 0
    grid_rate = {}
    grid_num = {}
    for (a, b) in X:

        grid_pos = min(9, int(a)) + (min(9, int(b)) * 10)

        if grid_pos not in grid_rate:
            grid_rate[grid_pos] = 0
            grid_num[grid_pos] = 0

        grid_rate[grid_pos] = grid_rate[grid_pos] + input_labels[label_offset]
        grid_num[grid_pos] = grid_num[grid_pos] + 1
        label_offset = label_offset + 1

    avg_label = np.mean(input_labels)

    lookup_table = {}
    lookup_table[-1] = avg_label
    
    pca_features = []
    for row in X:
        grid_pos = min(9, int(a)) + (min(9, int(b)) * 10)

        net_rate = grid_rate[grid_pos] / grid_num[grid_pos]

        if grid_num[grid_pos] < 50:
            net_rate = avg_label
        

        lookup_table[grid_pos] = net_rate
        pca_features.append(net_rate)

    pickle.dump(lookup_table, open(save_model_dir + "lookup_table_" + str(observed_days) + "_" + str(feature_offset), 'wb'))


    return pca_features

def predictArb(output_set, observed_days, output_labels):

    regression = pickle.load(open(save_model_dir + "logistic_regression" + str(observed_days), 'rb'))
    actual = pickle.load(open(save_model_dir + "actual" + str(observed_days), 'rb'))
    predict = pickle.load(open(save_model_dir + "predict" + str(observed_days), 'rb'))

    print "R2", r2_score(actual, predict)

    residuals = actual - predict 

    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals) 

    feature_id = singular_feature_num

    output_set[feature_id] = lookupPCAFeatures(output_set, feature_id, observed_days)
    feature_id = feature_id + 1

    for i in range(training_days):

        feature_ids = []
        for j in range(impact_num):
            for k in range(currency_num):
                feature_ids.append(featureId(i, j, k))

        output_set[feature_id] = lookupPCAFeatures(output_set[feature_ids], feature_id, observed_days)
        feature_id = feature_id + 1


    for j in range(impact_num):

        feature_ids = []
        for i in range(training_days):
            for k in range(currency_num):
                feature_ids.append(featureId(i, j, k))

        output_set[feature_id] = lookupPCAFeatures(output_set[feature_ids], feature_id, observed_days)
        feature_id = feature_id + 1

    for k in range(currency_num):

        feature_ids = []
        for i in range(training_days):
            for j in range(impact_num):
                feature_ids.append(featureId(i, j, k))

        output_set[feature_id] = lookupPCAFeatures(output_set[feature_ids], feature_id, observed_days)
        feature_id = feature_id + 1

    predict = regression.predict(output_set)

    residuals = output_labels - predict

    resid = (residuals[-1] - mean_resid) / std_resid

    return [resid, residuals]

def create_training_set(times, calendar):


    training_set = []

    for time in times:

        day = datetime.fromtimestamp(time).strftime("%A")

        if day == 'Saturday' or day == 'Sunday':
            continue

        feature_vector = [0] * singular_feature_num

        start_time = skipBack(time - (60 * 60 * 24 * calendar_backdays))
        end_time = time

        feature_vector = findCalendarFeatures(start_time, end_time, calendar, feature_vector)

        training_set.append(feature_vector)

    return training_set

def calcArb(pair, training_set, times, calendar):

    import time
    from datetime import datetime
    calendar_samples = 31536000
    rates_samples = 5000

    chosen_currency_pair = pair

    save_model_dir = "/tmp/" + chosen_currency_pair + "_"

    train_model = True
    if os.path.isfile(save_model_dir + "last_update"):
        prev_time = pickle.load(open(save_model_dir + "last_update", 'rb'))

        if (curr_time - prev_time) < 60 * 60 * 24 * 7:
            train_model = False

    train_model = True
    if train_model == False:
        rates_samples = 24 * calendar_backdays * 2
    else:
        print "Training Model", pair

    print "Downloading Rates"
    rates = pd.DataFrame.from_records(download_rates(chosen_currency_pair, rates_samples), columns=['time', 'rate'])

    print "last_rate", rates['rate'].tolist()[-1]

    rates.set_index('time', inplace=True)

    labels = {}
    actual_change = {}
    last_start_time = {}
    for observed_days in range(total_observed_days):
        labels[observed_days] = []
        last_start_time[observed_days] = []
        actual_change[observed_days] = []

    price_series = []

    for time in times:

        price_series.append(rates[(rates.index >= time)]['rate'].iloc[0])

        day = datetime.fromtimestamp(time).strftime("%A")

        if day == 'Saturday' or day == 'Sunday':
            continue

        end_time = time

        for observed_days in range(total_observed_days):
            start_time = skipBack(time - (60 * 60 * 12 * (observed_days + 1)))

            perc_change = calcPercChange(start_time, end_time, rates)
            labels[observed_days].append(perc_change)
            last_start_time[observed_days] = start_time
            

    pickle.dump(price_series, open(save_model_dir + "price_series", 'wb'))


    global_residuals = []
    for observed_days in range(total_observed_days):
        
        actual = labels[observed_days]
        
        if train_model == True:

            feature_id = singular_feature_num

            import time
            pickle.dump(int(time.time()), open(save_model_dir + "last_update", 'wb'))

            output_set = pd.DataFrame.from_records(training_set, columns=range(singular_feature_num))

            if len(actual) != len(training_set):
                print "size miss"
                sys.exit(0)

            # the whole thing
            output_set[feature_id] = createPCAFeatures(output_set, actual, feature_id, observed_days, 10)
            feature_id = feature_id + 1

            # then slice
            for i in range(training_days):

                feature_ids = []
                for j in range(impact_num):
                    for k in range(currency_num):
                        feature_ids.append(featureId(i, j, k))

                output_set[feature_id] = createPCAFeatures(output_set[feature_ids], actual, feature_id, observed_days)
                feature_id = feature_id + 1
                

            for j in range(impact_num):

                feature_ids = []
                for i in range(training_days):
                    for k in range(currency_num):
                        feature_ids.append(featureId(i, j, k))

                output_set[feature_id] = createPCAFeatures(output_set[feature_ids], actual, feature_id, observed_days)
                feature_id = feature_id + 1

            for k in range(currency_num):

                feature_ids = []
                for i in range(training_days):
                    for j in range(impact_num):
                        feature_ids.append(featureId(i, j, k))

                output_set[feature_id] = createPCAFeatures(output_set[feature_ids], actual, feature_id, observed_days)
                feature_id = feature_id + 1

            regression = linear_model.LinearRegression()
            regression.fit(output_set, actual)
            predict = regression.predict(output_set)

            pickle.dump(regression, open(save_model_dir + "logistic_regression" + str(observed_days), 'wb'))
            pickle.dump(actual, open(save_model_dir + "actual" + str(observed_days), 'wb'))
            pickle.dump(predict, open(save_model_dir + "predict" + str(observed_days), 'wb'))    

            # make sure revert and follow give the same signal for direction
            residuals = -predict 


            print "R2", r2_score(actual, predict)

            mean_resid = np.mean(residuals)
            std_resid = np.std(residuals)

            resid = (residuals[-1] - mean_resid) / std_resid

            pickle.dump([(r - mean_resid) / std_resid for r in residuals], open(save_model_dir + "residuals_future" + str(observed_days), 'wb'))


        else:
            [resid, residuals] = predictArb(pd.DataFrame.from_records(training_set, columns=range(singular_feature_num)), observed_days, actual)


        # transforms percentages to price 
        price = rates[(rates.index >= last_start_time[observed_days])]['rate']

        avg_price = np.mean(price)
        std_price = np.std(price)

        resid_price = residuals[-1]
        

        upper_price = price.iloc[len(price) - 1] + resid_price
        lower_price = price.iloc[len(price) - 1] - (resid_price / max(1, abs(resid)))

        global_residuals.append(resid)

        print "Final Residual: ", resid, (observed_days + 1), "days ", upper_price, "(TP)", lower_price, "(SL)"

    mean_resid = np.mean(global_residuals)
        
    print "******** Average Residual", mean_resid, chosen_currency_pair, "********"

    return mean_resid


currency_pairs = {
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
}


    
    

class CurrencyResid:

    def __init__(self):
        self.pair = ""
        self.resid = 0

class Currencies:

    def __init__(self):
        self.currency_list = []
        self.currency = ""
        self.avg_resid = 0
        self.num = 0

currency_ranking = {}
for currency in currencies:
    currency_ranking[currency] = Currencies()

rates = pd.DataFrame.from_records(download_rates("EUR_USD", 5000), columns=['time', 'rate'])
rates.set_index('time', inplace=True)

calendar_samples = 31536000

calendar_update = True
if os.path.isfile("calendar_update_" + str(calendar_samples)):
    prev_time = pickle.load(open("calendar_update_" + str(calendar_samples), 'rb'))

    if (curr_time - prev_time) < 60 * 60 * 2:
        calendar_update = False


if calendar_update == True:
    print "Downloading Calendar"
    calendar = pd.DataFrame.from_records(download_calendar(calendar_samples), columns=['currency', 'impact', 'actual', 'forecast', 'time', 'region'])
    pickle.dump(calendar, open("calendar_" + str(calendar_samples), 'wb'))
    pickle.dump(int(time.time()), open("calendar_update_" + str(calendar_samples), 'wb'))
else:
    calendar = pickle.load(open("calendar_" + str(calendar_samples), 'rb'))

calendar.set_index('time', inplace=True)

training_set = create_training_set(rates.index, calendar)

print len(training_set), "training_set_size"

for pair in currency_pairs:
    currency1 = pair[0:3]
    currency2 = pair[4:7]

    observation = CurrencyResid()
    observation.resid = calcArb(pair, training_set, rates.index, calendar)
    observation.pair = pair

    currency_ranking[currency1].currency = currency1
    currency_ranking[currency2].currency = currency2

    currency_ranking[currency1].currency_list.append(observation)
    currency_ranking[currency2].currency_list.append(observation)

    currency_ranking[currency1].avg_resid = currency_ranking[currency1].avg_resid + observation.resid
    currency_ranking[currency2].avg_resid = currency_ranking[currency2].avg_resid - observation.resid

    currency_ranking[currency1].num = currency_ranking[currency1].num + 1
    currency_ranking[currency2].num = currency_ranking[currency2].num + 1


currency_ranking = currency_ranking.values()


currency_ranking.sort(key = lambda x: -abs(x.avg_resid / max(1, x.num)))

overall_currency_map = {}
for currency_set in currency_ranking:
    if currency_set.num == 0:
        continue

    print currency_set.currency, currency_set.avg_resid / currency_set.num

    currency_set.currency_list.sort(key = lambda x: -abs(x.resid))

    overall_currency_map[currency_set.currency] = currency_set.avg_resid / currency_set.num

    line = " ******** " + currency_set.currency + " " + str(currency_set.avg_resid / currency_set.num) + " ******** "
    for pair in currency_set.currency_list:
        line += "[" + pair.pair + ", " + str(pair.resid) + "] "

    print line

pickle.dump(overall_currency_map, open("/tmp/" + "trend_model", 'wb'))

