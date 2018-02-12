import os

from matplotlib.lines import Line2D
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from multi_column_label_encoder import MultiColumnLabelEncoder
import numpy as np
import pandas as pd


def sale_price_to_bin( boundaries, sale_price ):
    b = 0
    while boundaries[b] <= sale_price:
        b += 1

    return b - 1

converters = {
              'MasVnrType': lambda x: None if x == 'NA' else x,
              'Electrical': lambda x: None if x == 'NA' else x,
              'LotFrontage': lambda x: None if x == 'NA' else int( x ),
              'MasVnrArea': lambda x: 0 if x == 'NA' else int( x ),
              'GarageYrBlt': lambda x: 0 if x == 'NA' else int( x ),

              'BsmtFinSF1': lambda x: None if x == 'NA' else int( x ),
              'BsmtFinSF2': lambda x: None if x == 'NA' else int( x ),
              'BsmtUnfSF': lambda x: None if x == 'NA' else int( x ),
              'TotalBsmtSF': lambda x: None if x == 'NA' else int( x ),
              'BsmtFullBath': lambda x: None if x == 'NA' else int( x ),
              'BsmtHalfBath': lambda x: None if x == 'NA' else int( x ),
              'GarageCars': lambda x: None if x == 'NA' else int( x ),
              'GarageArea': lambda x: None if x == 'NA' else int( x ),
              }

numeric = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
nominal = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
ordinal = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']

reduced_numeric = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch']
reduced_nominal = ['MSSubClass', 'MSZoning', 'LotShape', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'SaleType', 'SaleCondition']
reduced_ordinal = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']

attributes_dt = reduced_nominal + reduced_nominal + reduced_ordinal
attributes_reg = numeric

house_data = pd.read_csv( 'train.csv', index_col = 0, na_filter = False, converters = converters )

test_data = pd.read_csv( 'test.csv', index_col = 0, na_filter = False, converters = converters )

# Filling missing values
house_data[['MasVnrArea']] = house_data[['MasVnrArea']].fillna( house_data[['MasVnrArea']].mean() )
house_data[['LotFrontage']] = house_data[['LotFrontage']].fillna( house_data[['LotFrontage']].mean() )
house_data[['MasVnrType']] = house_data[['MasVnrType']].fillna( house_data[['MasVnrType']].mode() )
house_data[['Electrical']] = house_data[['Electrical']].fillna( house_data[['Electrical']].mode() )

test_data[['MasVnrArea']] = test_data[['MasVnrArea']].fillna( house_data[['MasVnrArea']].mean() )
test_data[['LotFrontage']] = test_data[['LotFrontage']].fillna( house_data[['LotFrontage']].mean() )
test_data[['MasVnrType']] = test_data[['MasVnrType']].fillna( house_data[['MasVnrType']].mode() )
test_data[['Electrical']] = test_data[['Electrical']].fillna( house_data[['Electrical']].mode() )

test_data[['BsmtFinSF1']] = test_data[['BsmtFinSF1']].fillna( house_data[['BsmtFinSF1']].mean() )

test_data[['BsmtFinSF2']] = test_data[['BsmtFinSF2']].fillna( house_data[['BsmtFinSF2']].mean() )

test_data[['BsmtFinSF1']] = test_data[['BsmtFinSF1']].fillna( house_data[['BsmtFinSF1']].mean() )

test_data[['BsmtUnfSF']] = test_data[['BsmtUnfSF']].fillna( house_data[['BsmtUnfSF']].mean() )

test_data[['TotalBsmtSF']] = test_data[['TotalBsmtSF']].fillna( house_data[['TotalBsmtSF']].mean() )

test_data[['BsmtFullBath']] = test_data[['BsmtFullBath']].fillna( 0 )  # fillna( house_data[['BsmtFullBath']].mode() )

test_data[['BsmtHalfBath']] = test_data[['BsmtHalfBath']].fillna( 0 )  # fillna( house_data[['BsmtHalfBath']].mode() )

test_data[['GarageCars']] = test_data[['GarageCars']].fillna( 2 )  # fillna( house_data[['GarageCars']].mode() )
test_data[['GarageArea']] = test_data[['GarageArea']].fillna( house_data[['GarageArea']].mean() )

# test_data.to_csv('test_data_no_na.csv')


# print house_data[['BsmtFullBath']].mode()
# print house_data[['BsmtHalfBath']].mode()
# print house_data[['GarageCars']].mode()
# exit()
# print test_data.shape
zero_price = [0 for i in range( test_data.shape[0] )]
test_data['SalePrice'] = zero_price
# print test_data.shape

combined = pd.concat( [house_data, test_data] )

# print ( combined.shape )

# print ( combined.tail() )
# dummies = pd.get_dummies( combined, columns = nominal )
# print ( dummies.head() )
# print ( dummies.shape )

changed = MultiColumnLabelEncoder( columns = nominal ).fit_transform( combined )
# print ( changed.shape )

# print ( changed.tail() )

new_train = changed[:][ :house_data.shape[0]]
new_test = changed[:][house_data.shape[0]:]
# new_test.to_csv( '+++_test_delete.csv' )
# exit()
train_data = new_train.drop('SalePrice',1)


y= new_train.loc[:,'SalePrice']
folds = 5
total_error = 0


kf = KFold(n_splits = folds)
kf.get_n_splits(train_data)
for train_index, test_index in kf.split(train_data):
  X_train, X_test = train_data[train_index], train_data[test_index]
  y_train, y_test = y[train_index], y[test_index]
  # Fit nn regressor on training data , train data contains SalePrice (Should change)
  #xg = xgboost.XGBRegressor()
  #xg = xg.fit(X_train,y_train)
  regressor = RandomForestRegressor(n_estimators=150, min_samples_split=1)
  regressor.fit(X_train, y_train)

  predictions = regressor.predict( X_test )
  error = abs(pred - y_test)
  model_error = sum(error)/len(error)
  total_error += model_error
  #print('Absolute Mean Error Model: ')
  #print(model_error)
  print('Average Absolute Mean Error:')
  print(total_error/folds)

#print y

#regressor = RandomForestRegressor(n_estimators=150, min_samples_split=1)
#regressor.fit(train_data, y)
#print


#prediction =(regressor.predict(train_data))
#print prediction
#df= pd.DataFrame(prediction)
#df.to_csv('RFregressor.csv')


