'''
Apply Nearest Neighbor Regressor to Encoded data of Housing price
'''

from __future__ import print_function
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import xgboost


numeric = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
nominal = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal','MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
# Read in train and test data (Categorical data are encoded)
n_train_data = pd.read_csv('new_train.csv', index_col = 0)
n_test_data = pd.read_csv('new_train.csv', index_col = 0)

house_data = pd.read_csv( 'train.csv', index_col = 0, na_filter = False)

# Read in combined data (Categorical data rae not encoded)
combined = pd.read_csv('combined.csv', index_col = 0, na_filter=False)

#Dummify
dummies = pd.get_dummies( combined, columns = nominal )
#print(train_data.head())
#print(test_data.head())
#print(combined.head())
train_data_w_sale = dummies[ :house_data.shape[0]]
test_data = dummies[house_data.shape[0]:]
#print(dummies.head())

#delete sale price to use in training
train_data = train_data_w_sale.drop('SalePrice',1)
#print(train_data)



#Do K fold Validation on nnregressor
folds = 5
total_error = 0
'''
for n in range(1,11):

	print("For neighbors: ", end=" ")
	print(n, end="\n")
	kf = KFold(n_splits = folds)
	kf.get_n_splits(train_data)
	for train_index, test_index in kf.split(train_data):
		X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
		y_train, y_test = train_data_w_sale.iloc[train_index]['SalePrice'], train_data_w_sale.iloc[test_index]['SalePrice']
		# Fit nn regressor on training data , train data contains SalePrice (Should change)
		neigh = KNeighborsRegressor(n_neighbors = n, weights='distance')
		neigh.fit(X_train,y_train)
		pred = neigh.predict(X_test)
		error = abs(pred - y_test)
		model_error = sum(error)/len(error)
		total_error += model_error
		#print('Absolute Mean Error Model: ')
		#print(model_error)
	print('Average Absolute Mean Error:', end=" ")
	print(total_error/folds, end="\n")
'''

kf = KFold(n_splits = folds)
kf.get_n_splits(train_data)
for train_index, test_index in kf.split(train_data):
	X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
	y_train, y_test = train_data_w_sale.iloc[train_index]['SalePrice'], train_data_w_sale.iloc[test_index]['SalePrice']
	# Fit nn regressor on training data , train data contains SalePrice (Should change)
	xg = xgboost.XGBRegressor()
	xg = xg.fit(X_train,y_train)
	predictions = xg.predict( X_test )
	error = abs(pred - y_test)
	model_error = sum(error)/len(error)
	total_error += model_error
	#print('Absolute Mean Error Model: ')
	#print(model_error)
	print('Average Absolute Mean Error:', end=" ")
	print(total_error/folds, end="\n")
	

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    