# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:54:54 2019

@author: Marco
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import metrics

# sklearn ML
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =============================================================================
# Load data
# =============================================================================
soybean = pd.read_csv('ALL_Prices_soybean.csv', index_col = 'Date', parse_dates=True).sort_index()
# soybean price plot
soybean.plot(y='bean_settle')
# examine NA
soybean.isna().sum(axis=0)
# describe soybean
soybean.describe()

# =============================================================================
# feature engineering
# =============================================================================
# create DateTime features
soybean['wday'] = soybean.index.weekday
soybean['week'] = soybean.index.week
soybean['month'] = soybean.index.month
soybean['quarter'] = soybean.index.quarter
soybean['year'] = soybean.index.year

# convert categorical variables to dummies
soybean = pd.get_dummies(soybean, columns=['wday', 'week', 'month', 'quarter']) 
# drop duplicates
soybean.drop_duplicates(inplace=True)
# Market_Open = 1
data = soybean.loc[soybean['Market_Open'] == 1]

# =============================================================================
# Split dataset
# =============================================================================
train = data[data.index < '2018-01-01']
test = data[data.index >= '2018-01-01']

# create X_train/test, Y_train/test
x_train, y_train = train.drop('bean_settle', axis=1), train['bean_settle']
x_test, y_test = test.drop('bean_settle', axis=1), test['bean_settle']


# =============================================================================
# GridSearch 
# =============================================================================
# Ridgge
param_set = {'alpha': [1, 1e-1, 1e-2, 1e-3, 1e-4],
              'normalize': ['True', 'False']}
gsearch = GridSearchCV(estimator = Ridge(),
                       param_grid = param_set,
                       scoring = 'neg_mean_squared_error', cv=5)

# Random Forest
param_set = {'n_estimators': range(5, 30, 5)}
gsearch = GridSearchCV(estimator = RandomForestRegressor(),
                       param_grid = param_set,
                       scoring = 'neg_mean_squared_error', cv=5)

# XGBoost
param_set = {'eta': [0.1, 0.2, 0.3],
              'n_estimators': [50, 100, 150],
              'max_depth': [5, 10]}
gsearch = GridSearchCV(estimator = XGBRegressor(),
                       param_grid = param_set,
                       scoring='neg_mean_squared_error', cv=5)
       
# =============================================================================
# Fit and Cross_Validate
# =============================================================================
# fit 
gsearch.fit(x_train, y_train)
print(gsearch.best_params_, -gsearch.best_score_.round(2)) 

# cross validate
scores = cross_validate(gsearch.best_estimator_, x_train, y_train, cv=5,
                         scoring=('r2', 'neg_mean_squared_error'),
                         return_train_score=True)
print("train CV MSE: ", -scores['train_neg_mean_squared_error'].round(2))
print("validate CV MSE: ", -scores['test_neg_mean_squared_error'].round(2))
print("Train r2: ", scores['train_r2']) 
    
# =============================================================================
# Visualization
# =============================================================================
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 
y_train_hat = DataFrame(gsearch.best_estimator_.predict(x_train), index=x_train.index)
mse_train = metrics.mean_squared_error(y_train, y_train_hat[0])
mape_train = mean_absolute_percentage_error(y_train, y_train_hat[0])

plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(16,9))
plt.plot(y_train, label='truth')
plt.plot(y_train_hat, 'r-', label='prediction')
plt.ylabel('Price (cent/bushel)', fontsize=28)
plt.xticks(rotation=90)
plt.title('Soybean Futures - Train Set', fontsize=32)
plt.legend(loc='best')
plt.show()

# test set
y_test_hat = DataFrame(gsearch.best_estimator_.predict(x_test), index=x_test.index)
mse_test = metrics.mean_squared_error(y_test, y_test_hat[0])
mape_test = mean_absolute_percentage_error(y_test, y_test_hat[0])

plt.figure(figsize=(16,9))
plt.plot(y_test, label='truth')
plt.plot(y_test_hat, 'r-', label='prediction')
plt.ylabel('Price (cent/bushel)', fontsize=28)
plt.xticks(rotation=90)
plt.title('Soybean Futures - Test Set', fontsize=32)
plt.legend(loc='best')
plt.show()

print("Train MSE: {:.2f} and MAPE: {:.2f}%".format(mse_train, mape_train))
print("Test MSE: {:.2f} and MAPE: {:.2f}%".format(mse_test, mape_test))






# =============================================================================
# Record performance
# =============================================================================
model_set = [Ridge(), RandomForestRegressor(), XGBRegressor()]
param_set = [
             {'alpha': 10.0**-np.arange(0,5,1), 
              'normalize': ['True', 'False']},              
             {'n_estimators': range(2, 12, 2),
              'max_depth': range(1, 6, 2),
              'max_features': ['auto', 'sqrt', 'log2']},
             {'eta': np.arange(1,4,1)/10,
              'n_estimators': np.arange(1,4,1)*50,
              'max_depth': range(1, 6, 2)}
             ]
eval_metric = 'neg_mean_squared_error'
fold = 5
array = []
for k, model in enumerate(model_set):
    print("\n", model, "\n")
    # GridSearchCV
    GSCV = GridSearchCV(estimator = model,
                        param_grid = param_set[k],
                        scoring = eval_metric, cv=fold)    
    GSCV.fit(x_train, y_train)
    print(GSCV.best_params_, -GSCV.best_score_.round(2))
    
    # Cross_Validate
    scores = cross_validate(GSCV.best_estimator_, x_train, y_train, cv=fold,
                            scoring=('r2', 'neg_mean_squared_error'),
                            return_train_score=True)    
    print("train CV MSE: ", -scores['train_neg_mean_squared_error'].round(2))
    print("validate CV MSE: ", -scores['test_neg_mean_squared_error'].round(2))
    print("Train r2: ", scores['train_r2'])  

    # Train vs Test
    y_train_hat = DataFrame(GSCV.best_estimator_.predict(x_train), index=x_train.index)
    mse_train = metrics.mean_squared_error(y_train, y_train_hat[0])
    mape_train = mean_absolute_percentage_error(y_train, y_train_hat[0])
    
    y_test_hat = DataFrame(GSCV.best_estimator_.predict(x_test), index=x_test.index)
    mse_test = metrics.mean_squared_error(y_test, y_test_hat[0])
    mape_test = mean_absolute_percentage_error(y_test, y_test_hat[0])
    print("Train MSE: {:.2f} and MAPE: {:.2f}%".format(mse_train, mape_train))
    print("Test MSE: {:.2f} and MAPE: {:.2f}%".format(mse_test, mape_test))

    array.append({'best estimator': GSCV.best_params_, 'best score': -GSCV.best_score_.round(2),
                  'train CV mean': -scores['train_neg_mean_squared_error'].round(2).mean(), 'train CV std': -scores['train_neg_mean_squared_error'].round(2).std(),
                  'val CV mean': -scores['test_neg_mean_squared_error'].round(2).mean(), 'val CV std': -scores['test_neg_mean_squared_error'].round(2).std(),
                  'train mse': mse_train, 'train mape': mape_train,
                  'test mse': mse_test, 'test mape': mape_test})
    
    y_train_hat = DataFrame(GSCV.best_estimator_.predict(x_train), index=x_train.index)
    mse_train = metrics.mean_squared_error(y_train, y_train_hat[0])
    mape_train = mean_absolute_percentage_error(y_train, y_train_hat[0])
    
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(16,9))
    plt.plot(y_train, label='truth')
    plt.plot(y_train_hat, 'r-', label='prediction')
    plt.ylabel('Price (cent/bushel)', fontsize=28)
    plt.xticks(rotation=90)
    plt.title('Soybean Futures - Train Set', fontsize=32)
    plt.legend(loc='best')
    plt.show()
    
    # test set
    y_test_hat = DataFrame(GSCV.best_estimator_.predict(x_test), index=x_test.index)
    mse_test = metrics.mean_squared_error(y_test, y_test_hat[0])
    mape_test = mean_absolute_percentage_error(y_test, y_test_hat[0])
    
    plt.figure(figsize=(16,9))
    plt.plot(y_test, label='truth')
    plt.plot(y_test_hat, 'r-', label='prediction')
    plt.ylabel('Price (cent/bushel)', fontsize=28)
    plt.xticks(rotation=90)
    plt.title('Soybean Futures - Test Set', fontsize=32)
    plt.legend(loc='best')
    plt.show()


df = DataFrame(data=array, index=['Ridge', 'Random Forest', 'XGB'])    
