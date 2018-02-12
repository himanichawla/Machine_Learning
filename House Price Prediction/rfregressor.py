'''
Created on Mar 16, 2017

@author: Manujinda Wathugala
Binning the data into different number of bins
and training regression models and checking their accuracy.
'''
import os

from matplotlib.lines import Line2D
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from multi_column_label_encoder import MultiColumnLabelEncoder
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

# print new_train.shape
# print new_test.shape

# new_train.to_csv( 'new_train.csv' )
# new_test.to_csv( 'new_test.csv' )

plot_dir = os.path.join( os.getcwd(), 'bin_distributions' )
if not os.path.exists( plot_dir ):
    os.mkdir( plot_dir )

sorted_hp = sorted( house_data['SalePrice'] )
samples = house_data.shape[0]
folds = 5

decision_tree_bin_errors = []
combined_bin_min_errors = []
combined_bin_model_errors = []
combined_bin_no_model_errors = []
combined_bin_no_model_min_errors = []


kf = KFold( n_splits = folds )
kf.get_n_splits( new_train )
new_train = new_train.reset_index( drop = True )
for train_index, test_index in kf.split( new_train ):

    X_train, X_test = new_train.iloc[train_index][nominal], new_train.iloc[test_index][nominal]
    y_train, y_test = new_train[train_index]['SalePrice'], new_train[test_index]['SalePrice']
    #ridgereg = Ridge(alpha=alpha,normalize=True)
    #ridgereg.fit(data[predictors],data['y'])
    #y_pred = ridgereg.predict(data[predictors])

    dt = Ridge(alpha=alpha,normalize=True)
    rgr = dt.fit(X_train,y_train)

    #dt = dt.fit( X_train, y_train )
    predictions = dt.predict( X_test )

    prediction_t_or_f = predictions == y_test
    correct = prediction_t_or_f[prediction_t_or_f]
    error = 1 - float( len( correct ) ) / len( predictions )
    print 'Error Decision Tree fold = {}'.format( error )

    total_error += error

total_error /= folds
print 'Error Decision Tree = {}'.format( total_error )


