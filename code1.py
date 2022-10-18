# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:23:18 2022

@author: aashutosh.bhattad
"""

import webbrowser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

import os
import logging
import glob
import time
import pandas_ta as ta
from nsepy import get_history


#Fetching Historical Data
data = get_history(symbol="NIFTY",
                            start=date(2010,1,1),
                            end=date(2022,8,27),
                            index=True)


data.rename(columns = {'High':'high_nifty', 'Low':'low_nifty',
                       'Close':'close_nifty','Open':'open_nifty'}, inplace=True)

#Feature Engineering

# Nifty Changes
data['nifty_close_next_day'] = (data['close_nifty'].shift(-1) - data['close_nifty'])*100/ data['close_nifty']
data['nifty_open_next_day'] = (data['open_nifty'].shift(-1) - data['close_nifty'])*100/ data['close_nifty']

data['nifty_high_next_day'] = (data['high_nifty'].shift(-1) - data['close_nifty'])*100/ data['close_nifty']
data['nifty_low_next_day'] = (data['close_nifty'] - data['low_nifty'].shift(-1))*100/ data['close_nifty']

data['nifty_day_gain'] = (data['close_nifty'] - data['open_nifty'])*100/data['open_nifty']
data['nifty_day_chng'] = (data['close_nifty'].shift(1) - data['close_nifty'])*100/data['close_nifty']
data['nifty_day_range'] = (data['high_nifty'] - data['low_nifty'])*100/data['low_nifty']

data['nifty_next_day_range'] = data['nifty_day_range'].shift(-1)
data['nifty_next_day_chng'] = data['nifty_day_chng'].shift(-1)
data['nifty_next_day_gain'] = data['nifty_day_gain'].shift(-1)

#wigs and side rallies
data['upside_rally'] = (data['high_nifty'] - data['open_nifty'])*100/data['open_nifty']
data['downside_rally'] = (data['open_nifty'] - data['low_nifty'])*100/data['open_nifty']
data['upside_wig'] = (data['high_nifty'] - data['close_nifty'])*100/data['close_nifty']
data['downside_wig'] = (data['close_nifty'] - data['low_nifty'])*100/data['close_nifty']

print(data.shape)
data.reset_index(inplace = True)
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data['DayofWeek'] = data['Date'].dt.dayofweek
data['WeekofYear'] = data['Date'].dt.isocalendar().week

print(data.shape)
data.isnull().sum()
data.dropna(inplace = True)
#Feature Selection


import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost
from xgboost import XGBClassifier , plot_importance, XGBRegressor

#Feature Engine based Selection
from feature_engine.selection import (
    DropDuplicateFeatures,
    DropConstantFeatures,
    DropDuplicateFeatures,
    DropCorrelatedFeatures,
    SmartCorrelatedSelection,
    SelectByShuffling,
    SelectBySingleFeaturePerformance,
    RecursiveFeatureElimination,
    DropHighPSIFeatures,
    RecursiveFeatureAddition,
)


#Train Test Split
y_vars = ['nifty_close_next_day','nifty_open_next_day','Date','nifty_high_next_day','nifty_low_next_day',
       'nifty_next_day_range', 'nifty_next_day_chng','nifty_next_day_gain']

#Train test Splits
import datetime
cut_off_date = datetime.datetime(2021,11, 1)

data['Date'] = pd.to_datetime(data['Date'], format = '%Y-%m-%d')
test = data[data['Date'] >= cut_off_date]
train = data[data['Date'] < cut_off_date]
print(train.shape)
print(test.shape)

tgt = 'nifty_close_next_day'

train_x = train.drop(y_vars, axis = 1)
train_y = train[tgt]

test_x = test.drop(y_vars, axis = 1)
test_y = test[tgt]

#Dropconstant features
constant = DropConstantFeatures(tol=0.05) #100% Values are same

# finds the constant features on the train set
constant.fit(train_x)
print(len(constant.features_to_drop_))

#retain Dayofweek
#Transforming
train_x = constant.transform(train_x)
test_x = constant.transform(test_x)
print(train_x.shape, test_x.shape)

#Correlation
smart_corr = SmartCorrelatedSelection(
    variables=None, # examines all variables
    method="pearson", # the correlation method
    threshold=0.8, # the correlation coefficient threshold
    missing_values="ignore",
    selection_method="model_performance", # how to select the features
    estimator= RandomForestRegressor(random_state=1, n_jobs = -1), 
)

# find correlated features and select the best from each group
# the method builds a random forest using each single feature from the correlated feature group
# and retains the feature from the group with the best performance

smart_corr.fit(train_x, train_y)
len(smart_corr.features_to_drop_)

train_x = smart_corr.transform(train_x)
test_x = smart_corr.transform(test_x)
print(train_x.shape, test_x.shape)

#sfe
sel = SelectBySingleFeaturePerformance(
    estimator = RandomForestRegressor(), # the model
    scoring="neg_root_mean_squared_error", # the metric to determine model performance
    cv=3, # the cross-validation fold,
# the performance threshold
)

sel.fit(train_x, train_y)
print(len(sel.features_to_drop_))

train_x = sel.transform(train_x)
test_x = sel.transform(test_x)
print(test_x.shape)

#Rfe

rfe = RecursiveFeatureElimination(
    estimator = RandomForestRegressor(), # the model
    scoring= "neg_root_mean_squared_error", # the metric to determine model performance
    cv=3, # the cross-validation fold
)

rfe.fit(train_x, train_y)

# plot of feature importance, derived from the Random Forests
pd.Series(rfe.feature_importances_).plot.bar(figsize=(20,5))
plt.ylabel('Feature importance derived from the random forests')
plt.show()

# same as above in a plot

pd.Series(rfe.performance_drifts_).sort_values().plot.bar(figsize=(20,5))
plt.ylabel('change in performance when removing feature')
plt.show()

rfe.features_to_drop_

train_x = rfe.transform(train_x)
test_x = rfe.transform(test_x)
print(test_x.shape)

test_x.columns

#Model Building

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(train_x, train_y)

preds = rf.predict(test_x)

from sklearn.metrics import mean_squared_error
from math import sqrt
sqrt(mean_squared_error(test_y, preds))

preds2 = rf.predict(train_x)
sqrt(mean_squared_error(train_y, preds2))


#XGBoost

eval_set = [(test_x, test_y)]
xgb = XGBRegressor(n_estimators=35, objective='reg:squarederror',
                    nthread=1,colsample_bytree =  0.3, gamma=  2,
                    learning_rate = 0.1, max_depth = 2, min_child_weight = 3,
                    early_stopping_rounds=10, eval_metric ='rmse')

xgb.fit(train_x, train_y,eval_set=eval_set, verbose=True)
preds1 = xgb.predict(test_x)
preds2 = xgb.predict(train_x)

print(sqrt(mean_squared_error(train_y, preds2)))
print(sqrt(mean_squared_error(test_y, preds1)))

xgb.get_params()
plot_importance(xgb)

test_df = pd.DataFrame()
test_df['date'] = test['Date']
test_df['actual'] = test_y
test_df['predicted'] = preds1

test_df.to_csv('eval_test.csv')
