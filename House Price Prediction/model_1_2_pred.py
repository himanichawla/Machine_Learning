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
from sklearn.ensemble import RandomForestClassifier
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
'''
new_test[['BsmtFinSF1']] = test_data[['BsmtFinSF1']].fillna( house_data[['BsmtFinSF1']].mean() )
new_test[['BsmtFinSF2']] = test_data[['BsmtFinSF2']].fillna( house_data[['BsmtFinSF2']].mean() )
new_test[['BsmtFinSF1']] = test_data[['BsmtFinSF1']].fillna( house_data[['BsmtFinSF1']].mean() )
new_test[['BsmtUnfSF']] = test_data[['BsmtUnfSF']].fillna( house_data[['BsmtUnfSF']].mean() )
new_test[['TotalBsmtSF']] = test_data[['TotalBsmtSF']].fillna( house_data[['TotalBsmtSF']].mean() )
new_test[['BsmtFullBath']] = test_data[['BsmtFullBath']].fillna( house_data[['BsmtFullBath']].mode() )
new_test[['BsmtHalfBath']] = test_data[['BsmtHalfBath']].fillna( house_data[['BsmtHalfBath']].mode() )
new_test[['GarageCars']] = test_data[['GarageCars']].fillna( house_data[['GarageCars']].mode() )
new_test[['GarageArea']] = test_data[['GarageArea']].fillna( house_data[['GarageArea']].mean() )
'''
# print(new_test.iloc[0])



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
reg_models = []
dec_tree = 0
# Try different numbers of bins
for bins in range( 3, 4 ):

    bin_data = []

    lo_index = 0
    hi_index = 0

    print ( '\nTotal bins to create : {}'.format( bins ) )
    tot = 0
    boundaries = [0]
    last_bin = False

    # Bin samples in to bins
    for b in range( bins ):

        # Remaining samples yet to be binned
        remaining_samples = samples - tot
        # Remaining number of bins
        remaining_bins = bins - b

        # Frequency for the remaining bins.
        freq = int( remaining_samples / remaining_bins )
        # Excess samples that needs to be evenly distributed
        # among the bins. The pigeon hole principle.
        excess_samples = remaining_samples % remaining_bins

        # If there are excess samples, increase the bin frequency
        # if b < rem:
        if excess_samples != 0:
            hi_index += freq + 1
        else:
            hi_index += freq

        if hi_index >= samples:
            # This is the last bin.
            # Put all the remaining examples in this bin
            hi_index = samples - 1

            # Since this is the last bin we need to include
            # samples with target value sorted_hp[hi_index]
            # in this bin.
            # Since our criteria to select samples for a bin is
            # 'SalePrice < hi, add 1 to sorted_hp[hi_index]
            # so that we can include these sample in the last bin
            hi = sorted_hp[hi_index] + 1
            last_bin = True
        else:
            # Borderline value for this bin.
            # hi will fall in to the next bin
            hi = sorted_hp[hi_index]

            # We do not want samples with the same target value
            # to be in two bins.

            # Count the number of samples with indexes >= hi_index
            # with the same hi value
            up = 1
            while hi_index + up < samples and sorted_hp[hi_index + up] == hi:
                up += 1

            # Count the number of samples with indexes < hi_index
            # with the same hi value
            down = 1
            while hi_index - down >= 0 and sorted_hp[hi_index - down] == hi:
                down += 1
            down -= 1

            # Decide whether to put all the samples with the target
            # value hi in this bin or the next bin.
            if down > up:
                # We already have more samples with the target value hi
                # in this bin. So include all the samples with target value
                # hi in this bin
                hi_index += up
            else:
                # More samples with the target value hi falls in the next bin
                if hi_index - down - 1 <= lo_index:
                    # This bin will be empty. So include all the samples with
                    # target value hi in this bin.
                    hi_index += up
                else:
                    # Include all the samples with target value hi
                    # in the next bin.
                    hi_index -= down

            # After deciding which bin the multiple copies
            # go into, check whether we are at the last bin.
            if hi_index >= samples:
                hi_index = samples - 1
                hi = sorted_hp[hi_index] + 1
                last_bin = True
            else:
                # Borderline value for this bin after adjusting for
                # sample with the same target value.
                # hi will fall in to the next bin
                hi = sorted_hp[hi_index]


#         hi = sorted_hp[hi_index]
#         hi = hi + 1 if last_bin else hi
        lo = sorted_hp[lo_index]
        # print ( lo_index, hi_index )

        temp_df = new_train[( new_train.SalePrice >= lo ) & ( new_train.SalePrice < hi )]
        temp_df_reset = temp_df.reset_index( drop = True )
        bin_data.append( temp_df_reset )

        boundaries.append( hi )
        lo_index = hi_index

        # print ( bin_data[b].shape )
        tot += bin_data[b].shape[0]

        if last_bin:
            bins = b + 1
            break

        # Create bin histograms
#         plt.hist( bin_data[b]['SalePrice'], 10 )
#
#         plt.title( 'Sale Price Distribution - Bins - {} Bin {}'.format( bins, b ) )
#         plt.ylabel( 'Number of houses' )
#         plt.xlabel( 'Sale Prices' );
#
#         plt.savefig( os.path.join( plot_dir, '{}_Bins_Bin_{}.png'.format( bins, b ) ) )
#         # plt.show()
#         plt.close()
#         plt.cla()
#         plt.clf()

    # print ( tot )
    print ( 'Actual bins created  : {}'.format( bins ) )
    # print ( boundaries )

    bin_labels = pd.Series( [sale_price_to_bin( boundaries, sp ) for sp in new_train['SalePrice']] )

    # print ( bin_labels[:10] )

    partition_bin = []
    for b in range( bins ):
        bin_part = []
        X_train = bin_data[b]
        y_train = pd.Series( [sale_price_to_bin( boundaries, sp ) for sp in X_train['SalePrice']] )

        for f in range( folds, 1, -1 ):
            percent_test = 1.0 / f
            X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = percent_test )
            bin_part.append( ( X_test, y_test ) )

        bin_part.append( ( X_train, y_train ) )

        partition_bin.append( bin_part )

    # print ( len( partition_bin ) )
    # print ( len( partition_bin[0] ) )
    # print ( partition_bin[0][0][0] )
    # print ( partition_bin[1][0][1] )
    # exit()

    total_error_dt = 0
    errors_dt = []
    errors_reg = []
    errors_model = []
    tot_min_error_list = []
    tot_model_error_list = []
    tot_no_model_error_list = []
    tot_no_model_min_error_list = []

    predicted_bin_sizes = []

    for c in range( folds ):
        X_reg_train = []
        y_reg_train = []

        for b in range( bins ):
            df_train = pd.DataFrame()
            y_train = pd.Series()

            for c2 in range( folds ):
                if c2 != c:
                    df_train = df_train.append( partition_bin[b][c2][0] )
                    y_train = y_train.append( partition_bin[b][c2][1] )

            X_reg_train.append( df_train )
            y_reg_train.append( y_train )

        df_dt_train = pd.DataFrame()
        y_dt_train = pd.Series()

        df_dt_test = pd.DataFrame()
        y_dt_test = pd.Series()

        # Create training and validation sets for decision tree
        for b in range( bins ):
            df_dt_train = df_dt_train.append( X_reg_train[b] )
            y_dt_train = y_dt_train.append( y_reg_train[b] )

            df_dt_test = df_dt_test.append( partition_bin[b][c][0] )
            y_dt_test = y_dt_test.append( partition_bin[b][c][1] )

#         print '--------------'
#         print ( df_dt_train.shape )
#         print ( len( y_dt_train ) )
#         print ( df_dt_test.shape )
#         print ( len( y_dt_test ) )

        # Train decision tree for this fold
        # dt = tree.DecisionTreeClassifier()
        dt = RandomForestClassifier( n_estimators = 20 )


        # dt = dt.fit( df_dt_train[nominal], y_dt_train )
        # predictions = dt.predict( df_dt_test[nominal] )
        dt = dt.fit( df_dt_train[attributes_dt], y_dt_train )
        dec_tree = dt
        predictions = dt.predict( df_dt_test[attributes_dt] )

#         print ( df_dt_test.shape )
        df_dt_test['Pred'] = predictions
        dt_predicted_reg_test = df_dt_test.groupby( 'Pred' )
#         print grp.get_group( 0 )
#         print ( df_dt_test.shape )
#         # print predictions
#         exit()

        prediction_t_or_f = predictions == y_dt_test
        correct = prediction_t_or_f[prediction_t_or_f]
        error = 1 - float( len( correct ) ) / len( predictions )
        # print 'Error Decision Tree fold = {}'.format( error )

        errors_dt.append( error )

        total_error_dt += error

        total_error_reg = 0
        bin_error_reg = []
        bin_size_reg = []
        fold_error_reg = []
        fold_error_model = []
        fold_error_no_model = []
        fold_pred_bin_size = []
        tot_min_err = 0
        tot_min_err_size = 0
        tot_model_error = 0
        tot_no_model_error = 0
        tot_no_model_min_error = 0
        tot_model_error_size = 0
        for b in range( bins ):
            reg = linear_model.LinearRegression()
            reg.fit( X_reg_train[b][attributes_reg], X_reg_train[b]['SalePrice'] )
            reg_models.append( reg )

            pred = reg.predict( partition_bin[b][c][0][attributes_reg] )
            error = abs( pred - partition_bin[b][c][0]['SalePrice'] )
            reg_error = float( sum( error ) ) / len( error )
            fold_error_reg.append( reg_error )
            tot_min_err += sum( error )  # reg_error * len( error )
            tot_min_err_size += len( error )

            bin_mean = partition_bin[b][c][0]['SalePrice'].mean()
            no_model_min_errors = abs( bin_mean - partition_bin[b][c][0]['SalePrice'] )
            tot_no_model_min_error += sum( no_model_min_errors )

            try:
                pred_model = reg.predict( dt_predicted_reg_test.get_group( b )[attributes_reg] )
                error_model = abs( pred_model - dt_predicted_reg_test.get_group( b )['SalePrice'] )
                no_model_errors = abs( bin_mean - dt_predicted_reg_test.get_group( b )['SalePrice'] )

                len_error_model = len( error_model )
                sum_error_model = sum( error_model )
                sum_no_model_errors = sum( no_model_errors )
                model_error = float( sum_error_model ) / len_error_model
            except KeyError:
                len_error_model = 0
                sum_error_model = 0
                sum_no_model_errors = 0
                model_error = 0


            fold_error_model.append( model_error )
            fold_pred_bin_size.append( len_error_model )
            tot_model_error += sum_error_model  # model_error * len( error_model )
            tot_model_error_size += len_error_model
            tot_no_model_error += sum_no_model_errors




#             print ( partition_bin[b][c][1] )
#             print ( pred )
#             exit()


        errors_reg.append( fold_error_reg )
        errors_model.append( fold_error_model )
        predicted_bin_sizes.append( fold_pred_bin_size )

        tot_min_error_list.append( tot_min_err / tot_min_err_size )
        tot_model_error_list.append( tot_model_error / tot_model_error_size )
        tot_no_model_error_list.append( tot_no_model_error / tot_model_error_size )
        tot_no_model_min_error_list.append( tot_no_model_min_error / tot_model_error_size )

    print 'Errors DT:       ', errors_dt
    print
    print 'Tot min error:   ', tot_min_error_list
    print 'Tot model error: ', tot_model_error_list
    print
    print 'Errors Reg:      ', errors_reg
    print 'Errors Model:    ', errors_model
    # print len( errors_reg ), len( errors_reg[0] )
    # print len( predicted_bin_sizes ), len( predicted_bin_sizes[0] )

    bin_tot_min_err = []
    bin_tot_model_err = []

    for b in range( bins ):
        tot_min_err = 0
        tot_min_err_size = 0
        tot_model_err = 0
        tot_model_err_size = 0
        for f in range( folds ):
            tot_min_err += errors_reg[f][b] * len( partition_bin[b][f] )
            tot_min_err_size += len( partition_bin[b][f] )

            tot_model_err += errors_model[f][b] * predicted_bin_sizes[f][b]
            tot_model_err_size += predicted_bin_sizes[f][b]

        bin_tot_min_err.append( tot_min_err / tot_min_err_size )
        bin_tot_model_err.append( tot_model_err / tot_model_err_size )

    print
    print 'Binwise min error:   ', bin_tot_min_err
    print 'Binwise model error: ', bin_tot_model_err

    print
    print 'Combined dt error:    ', sum( errors_dt ) / len( errors_dt )
    print 'Combined min error:   ', sum( tot_min_error_list ) / len( tot_min_error_list )
    print 'Combined model error: ', sum( tot_model_error_list ) / len( tot_model_error_list )

    decision_tree_bin_errors.append( sum( errors_dt ) / len( errors_dt ) )
    combined_bin_min_errors.append( sum( tot_min_error_list ) / len( tot_min_error_list ) )
    combined_bin_model_errors.append( sum( tot_model_error_list ) / len( tot_model_error_list ) )
    combined_bin_no_model_errors.append( sum( tot_no_model_error_list ) / len( tot_no_model_error_list ) )
    combined_bin_no_model_min_errors.append( sum( tot_no_model_min_error_list ) / len( tot_no_model_min_error_list ) )

print
print decision_tree_bin_errors
print combined_bin_min_errors
print combined_bin_model_errors

'''
fig1 = plt.figure()
ax = fig1.add_subplot( 111 )

plt.axis( [0, len( combined_bin_min_errors ), 0, max( combined_bin_model_errors + combined_bin_no_model_errors ) + 10] )
line = Line2D( range( 1, len( combined_bin_min_errors ) + 1 ), combined_bin_min_errors, marker = 'o', markersize = 8, linestyle = '--', label = 'Minimum Error Possible' )
ax.add_line( line )
line2 = Line2D( range( 1, len( combined_bin_model_errors ) + 1 ), combined_bin_model_errors, color = 'r', marker = '*', markersize = 13, linestyle = '--', label = 'Model Accuracy' )
ax.add_line( line2 )

line3 = Line2D( range( 1, len( combined_bin_no_model_errors ) + 1 ), combined_bin_no_model_errors, color = 'g', marker = 'D', markersize = 8, linestyle = '--', label = 'Should do better' )
ax.add_line( line3 )

line4 = Line2D( range( 1, len( combined_bin_no_model_min_errors ) + 1 ), combined_bin_no_model_min_errors, color = 'c', marker = 's', markersize = 8, linestyle = '--', label = 'Should do better - min' )
ax.add_line( line4 )

# plt.xticks( range( 1, bins + 1 ) )



plt.xlabel( 'Number of Bins' )
plt.ylabel( 'Absolute Mean Error' )
plt.title( 'Absolute Mean Error Minimum Bound vs. Achieved' )
plt.legend( handles = [line, line2, line3, line4], loc = 'lower left' )

plt.show()

fig2 = plt.figure()
ax = fig2.add_subplot( 111 )

plt.axis( [0, len( decision_tree_bin_errors ), 0, max( decision_tree_bin_errors )] )
line5 = Line2D( range( 1, len( decision_tree_bin_errors ) + 1 ), decision_tree_bin_errors, marker = 'o', markersize = 8, linestyle = '--', label = 'Decision Tree Bin Prediction Error' )
ax.add_line( line5 )

plt.xlabel( 'Number of Bins' )
plt.ylabel( 'Error Ratio' )
plt.title( 'Decision Tree Bin Prediction Error' )
plt.legend( handles = [line5], loc = 'upper left' )

plt.show()
'''
train = new_test
# test data
print( new_train.shape )
print( new_test.shape )
'''
for row in train[attributes_dt].iterrows():
    # print row
    try:

        pred = dec_tree.predict( row )
    except ValueError as E:
        print( E )
       # print(row)
        exit()
exit()
'''
pred = dt.predict( train[attributes_dt] )
train['prediction'] = pred
test_bins = train.groupby( 'prediction' )
'''
for r in test_bins:
    print r

exit()
'''
# final_pred = np.array( [] )
predictions_df = pd.DataFrame()
for b in range( bins ):
    try:

        reg_pred = reg_models[b].predict( test_bins.get_group( b )[attributes_reg] )
        print b, len( reg_pred ), type( reg_pred )
        # print reg_pred
        # final_pred.append( reg_pred )
        # test_bins.get_group( b )['SalePrice'] = reg_pred

        # print type( test_bins.get_group( b ) )
        temp_df = test_bins.get_group( b )
        temp_df['SalePrice'] = reg_pred
        # print temp_df.shape
        predictions_df = predictions_df.append( temp_df )

        # final_pred = np.concatenate( ( final_pred, reg_pred ), axis = 0 )
    except:
        continue
        print 'Except'

print predictions_df.shape

predictions_df.sort_index( axis = 0 )
print ( predictions_df['SalePrice'] )
predictions_df.to_csv( '++Test_set_predictions.csv' )
exit()

for grp in test_bins.groups:

    # Number of examples in this group
    g_size = len( test_bins.get_group( grp ) )

    print test_bins.get_group( grp )
