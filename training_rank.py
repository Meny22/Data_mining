import numpy as np
import pandas as pd


import os
import pandas as pd
#import boto3
from io import StringIO
import io
import string
import random
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from pyltr.models.lambdamart import LambdaMART
from pyltr.metrics import NDCG

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:f}'.format})

def clustering(train):
    from sklearn.cluster import KMeans
    search_data = train[["srch_destination_id", "srch_length_of_stay", "srch_booking_window", "srch_adults_count",
                         "srch_children_count", "srch_room_count", "srch_saturday_night_bool"]]
    ##prop_data = train[["prop_country_id","prop_starrating","prop_review_score","prop_brand_bool","prop_location_score1","prop_location_score2","prop_log_historical_price","price_usd","promotion_flag"]]
    prop_data = train[
        ["prop_starrating", "prop_review_score", "prop_brand_bool", "prop_location_score1", "prop_location_score2",
         "prop_log_historical_price", "price_usd", "promotion_flag"]]
    # cluster_data = train[]
    # kmeans = KMeans(n_clusters=4)
    # kmeans.fit(train)
    # kmeans.labels_
    kmeans_search = KMeans(n_clusters=10)
    kmeans_prop = KMeans(n_clusters=10)

    kmeans_search.fit(search_data)
    kmeans_prop.fit(prop_data)
    train_copy = train.copy()
    train_copy["search_cluster_id"] = kmeans_search.labels_.tolist()
    train_copy["prop_cluster_id"] = kmeans_prop.labels_.tolist()
    return train_copy

def rank(x,y):
    from operator import itemgetter
    rank = np.zeros((len(x),3))
    for i,x_row in enumerate(x):
        x_row = x_row[0:3]
        x_row[2] = round(y[i][1],3)
        rank[i] = x_row
    rank.view('i8,i8,i8').sort(order=['f0','f2'], axis=0)
    rank = rank[:,:2]
    np.savetxt("foo.csv", rank, delimiter=",",fmt=['%d','%d','%.3f'],header="srch_id,prop_id")

def preprocess(data,training):
    new_df = data
    for i, name in enumerate(new_df.columns):
        missing_values = new_df[name].isnull().sum()
        percentage = missing_values / new_df['srch_id'].count()
        if percentage > 0.4:
            new_df = new_df.drop(columns=[name])

    new_df["prop_review_score"] = new_df["prop_review_score"].fillna(new_df["prop_review_score"].mean())
    new_df["prop_location_score2"] = new_df["prop_location_score2"].fillna(new_df["prop_location_score2"].mean())
    new_df["orig_destination_distance"] = new_df["orig_destination_distance"].fillna(
    new_df["orig_destination_distance"].mean())
    if training:
        new_df = new_df[["srch_id", "prop_id", "price_usd", "prop_location_score1", "prop_location_score2", "prop_review_score",
                "srch_room_count", "srch_children_count", "srch_adults_count", "srch_booking_window", "prop_starrating",
                "promotion_flag", "prop_brand_bool","click_bool", "booking_bool", "random_bool"]]
    else:
        new_df = new_df[
            ["srch_id", "prop_id", "price_usd", "prop_location_score1", "prop_location_score2", "prop_review_score",
             "srch_room_count", "srch_children_count", "srch_adults_count", "srch_booking_window", "prop_starrating",
             "promotion_flag", "prop_brand_bool", "random_bool"]]
    new_df["price_usd"] = (new_df["price_usd"] - new_df["price_usd"].min()) / (
                new_df["price_usd"].max() - new_df["price_usd"].min())

    return new_df

def predict(df_test,model):
    x_test = df_test.drop(["srch_id", "prop_id"], axis=1)
    if "target_score" in x_test:
        x_test.drop(["target_score"], axis=1, inplace=True)

    print("Predicting...")
    result = model.predict(x_test)

    preds = pd.DataFrame({
        "srch_id": df_test["srch_id"],
        "prop_id": df_test["prop_id"],
        "result": result
    })

    out(preds,"new.csv")

def out(df,out_file):
    df.sort_values(["srch_id", "result"], inplace=True, ascending=[True, False])
    df.drop("result", axis=1, inplace=True)

    print("Writing " + out_file + "...")
    df.to_csv(out_file, index=False)


train = pd.read_csv('training_set_VU_DM.csv')
test = pd.read_csv('test_set_VU_DM.csv')


def getmodel(grad,X_train,y_train,query_ids):
    if grad:
        model = GradientBoostingRegressor(n_estimators=100, verbose=1)
        model.fit(X_train, y_train)
    else:
        query_ids = query_ids.copy()
        model = LambdaMART(metric=NDCG(len(X_train)), n_estimators=100, verbose=1)
        model.fit(X_train,y_train,query_ids)
    return model


#train = new_df
#new_df = clustering(new_df)
train = preprocess(train,True)
click_indices = train[train.booking_bool == 1].index
random_indices = np.random.choice(click_indices, len(train.loc[train.booking_bool == 1]), replace=False)
click_sample = train.loc[random_indices]

not_click = train[train.booking_bool == 0].index
random_indices = np.random.choice(not_click, sum(train['booking_bool']), replace=False)
not_click_sample = train.loc[random_indices]

us_new = pd.concat([not_click_sample, click_sample], axis=0)

print("Percentage of not click impressions: ", len(us_new[us_new.booking_bool == 0])/len(us_new))
print("Percentage of click impression: ", len(us_new[us_new.booking_bool == 1])/len(us_new))
print("Total number of records in resampled data: ", len(us_new))
us_new["target_score"] = 0
us_new.loc[us_new["click_bool"] == 1, "target_score"] = 1
us_new.loc[us_new["booking_bool"] == 1, "target_score"] = 5
us_new.drop(["click_bool", "booking_bool"], axis=1, inplace=True)

#X=us_new.drop(['date_time','position',"click_bool","booking_bool"], 1).values
X=us_new
Y=us_new[["target_score"]]
X_seperated = []
y_seperated = []
print(X)

#X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
X_train = X
X_train.sort_values(by='srch_id', inplace=True)
y_train = Y
X_train_ = X_train.drop(["target_score", "srch_id", "prop_id"], axis=1)
#X_test_ = X_test.drop(["target_score", "srch_id", "prop_id"], axis=1)
model = getmodel(False,X_train_,y_train,X_train["srch_id"])
predict(preprocess(test,False),model)
#predicted_values = model.predict(X_test)
# print("Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values))
# print("---------------------------------------\n")

# test = preprocess(test,False)
# test = test.values
# predictions = grad.predict_proba(test)
# rank(test,predictions)
# print(data[0])
# X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
