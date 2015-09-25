#!/usr/bin/python

import sys
import signal
import time
import shelve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
import datetime
# import multiprocessing as mp


class MyTimeoutException(Exception):
    pass


def handler(signum, frame):
    print "Fit ran out of time!"
    raise MyTimeoutException("fit timeout")


def load_data(which):
    if which == 'newer':
        infile = '../data/BEA-RegionalIncomeByIndustry/CA5N_2001_2013_MSA.csv'
        industry_code = 500
        year_cols = [str(yr) for yr in range(2001, 2014)]
    elif which == 'older':
        infile = '../data/BEA-RegionalIncomeByIndustry/CA5_1969_2000_MSA.csv'
        industry_code = 400
        year_cols = [str(yr) for yr in range(1969, 2001)]

    df = pd.read_csv(infile, low_memory=False)

    # select a specific industry
    df = df[df['LineCode'] == industry_code]

    # utility function to remove characters from number strings
    def clean_data_entry(x):
        m = re.match("\d+", x)
        # i.e. - if the string starts with
        if m:
            return float(m.group())
        else:
            return np.nan

    df[year_cols] = df[year_cols].applymap(clean_data_entry)

    return df


def clean_nans(df, which):
    if which == 'newer':
        year_cols = [str(yr) for yr in range(2001, 2014)]
    elif which == 'older':
        year_cols = [str(yr) for yr in range(1969, 2001)]

    nan_indexes = pd.isnull(df[year_cols]).any(1).nonzero()[0]

    nan_df = df.iloc[nan_indexes]
    df.drop(df.iloc[nan_indexes].index, inplace=True)

    return df, nan_df


def relative_normalize(df, which):
    if which == 'newer':
        year_cols = [str(yr) for yr in range(2001, 2014)]
    elif which == 'older':
        year_cols = [str(yr) for yr in range(1969, 2001)]

    means_df = df[year_cols].mean()

    for year in year_cols:
        df[year] = df[year]/means_df[year]-1

    return df


def merge_dfs(older_df, newer_df):
    # change to index dataframes by geofips for consistent merging
    older_df.set_index('GeoFIPS', inplace=True, verify_integrity=True)

    newer_df.set_index('GeoFIPS', inplace=True, verify_integrity=True)

    # split off conflicting columns from the 2 dfs.
    conflicting_cols = ['GeoName',
                        'Region',
                        'Table',
                        'LineCode',
                        'IndustryClassification',
                        'Description']
    conflicting_col_older_df = older_df[conflicting_cols]
    conflicting_col_older_df.index = older_df.index
    which = 'Old'
    rename_cols = ['%sGeoName' % which,
                   '%sRegion' % which,
                   '%sTable' % which,
                   '%sLineCode' % which,
                   '%sIndustryClassificiation' % which,
                   '%sDescription' % which]
    conflicting_col_older_df.columns = rename_cols
    conflicting_col_newer_df = newer_df[conflicting_cols]
    conflicting_col_newer_df.index = newer_df.index
    which = 'New'
    rename_cols = ['%sGeoName' % which,
                   '%sRegion' % which,
                   '%sTable' % which,
                   '%sLineCode' % which,
                   '%sIndustryClassificiation' % which,
                   '%sDescription' % which]
    conflicting_col_newer_df.columns = rename_cols

    # now drop conflicting cols from the 2 df's
    newer_df.drop(conflicting_cols, axis=1, inplace=True)
    older_df.drop(conflicting_cols, axis=1, inplace=True)

    # merge conflicting dfs
    extras_merged_inner = pd.concat(
        [conflicting_col_older_df, conflicting_col_newer_df],
        axis=1, join='inner')

    # merge data part of dfs
    data_merged_inner = pd.concat([older_df, newer_df], axis=1, join='inner')

    # merge data back with other part for full df
    merged_df = pd.concat([extras_merged_inner, data_merged_inner], axis=1)

    return merged_df


def windowed_svr_fit_model(X, Y, ntest=0, window=5, ker='linear'):
    ''' creates and fits an SVR model to data.
    accepts np.arrays X and Y'''
    start_time = datetime.datetime.now()

    def getWindows(arr):
        numPts = len(arr)

        i = 0
        j = window
        x = []
        y = []
        while j < numPts:
            x.append(arr[i:j])
            i += 1
            j += 1

        j = window
        while j < numPts:
            y.append(arr[j])
            j += 1
        return x, y

    trans = StandardScaler()
    Y = trans.fit_transform(Y)

    if ntest != 0:
        Ytrain = Y[0:-ntest]
        Ytest = Y[-ntest:]

        Xtrain = X[0:-ntest]
        Xtest = X[-ntest:]
    else:
        Ytrain = Y
        Ytest = None
        Xtrain = X
        Xtest = None

    xtrain, ytrain = getWindows(Ytrain)

    Ytrain = trans.inverse_transform(Ytrain)
    if ntest != 0:
        Ytest = trans.inverse_transform(Ytest)

    # print 'ytrain',len(ytrain),ytrain
    # print 'xtrain',len(xtrain),xtrain

    # windowed svr gridsearch
    if(ker == 'rbf'):
        parameters = {'gamma': [2**i for i in range(-30, 30)],
                      'C': [2**i for i in range(-30, 30)]}
    elif(ker == 'linear'):
        parameters = {'C': [2**i for i in range(-20, 20)]}
    svr = SVR(kernel=ker)

    grid = GridSearchCV(svr, param_grid=parameters)
    fitted_svr = grid.fit(xtrain, ytrain).best_estimator_

    end_time = datetime.datetime.now()
    delta_time = end_time - start_time
    comp_time_sec = delta_time.total_seconds()  # seconds
    print 'comp_time:', comp_time_sec

    ret_dict = {'model': fitted_svr,
                'window': window,
                'transform': trans,
                'xtrain': xtrain,
                'ytrain': ytrain,
                'Xtrain': Xtrain,
                'Ytrain': Ytrain,
                'Xtest': Xtest,
                'Ytest': Ytest,
                'comp_time_sec': comp_time_sec}

    return ret_dict


def windowed_svr_get_result(args_dict):
    model = args_dict['model']
    window = args_dict['window']
    trans = args_dict['transform']
    xtrain = args_dict['xtrain']
    ytrain = args_dict['ytrain']
    Xtrain = args_dict['Xtrain']
    Ytrain = args_dict['Ytrain']
    Xtest = args_dict['Xtest']
    Ytest = args_dict['Ytest']
    comp_time_sec = args_dict['comp_time_sec']

    output_dict = {}
    output_dict['train_years'] = Xtrain
    output_dict['train_values'] = Ytrain

    def forecast_iteration(x, n_pred):
        if n_pred > 1:
            y = model.predict(x)
            y_curr = [y]
            next_x = np.append(x[1:], y_curr)
            y_next_deeper = forecast_iteration(next_x, n_pred-1)
            y_previous_shallower = np.append(y_curr, y_next_deeper)
            return y_previous_shallower
        else:
            y = model.predict(x)
            return [y]

    ypred_train = model.predict(xtrain)
    train_score = r2_score(ytrain, ypred_train)
    print 'train score:', train_score

    if Ytest is not None:
        ypred_test = forecast_iteration(Ytrain[-window:], len(Ytest))
        test_score = r2_score(Ytest, ypred_test)
        print ' test score:', test_score

    pred_len = 20
    longer_pred = forecast_iteration(Ytrain[-window:], pred_len)
    Xlonger = [yr for yr in range(Xtrain[-1] + 1, Xtrain[-1] + 1 + pred_len)]

    ytrain = trans.inverse_transform(ytrain)
    ypred_train = trans.inverse_transform(ypred_train)
    longer_pred = trans.inverse_transform(longer_pred)

    output_dict['train_pred'] = ypred_train
    output_dict['train_score'] = train_score
    output_dict['proj_years'] = Xlonger
    output_dict['proj_pred'] = longer_pred

    if Ytest is not None:
        ypred_test = trans.inverse_transform(ypred_test)
        output_dict['test_years'] = Xtest
        output_dict['test_values'] = Ytest
        output_dict['test_pred'] = ypred_test
        output_dict['test_score'] = test_score

    plt.plot_date(Xtrain, Ytrain, '-o', color='green')
    plt.plot_date(Xtrain[window:], ypred_train, '--', color='red')
    if Ytest is not None:
        plt.plot_date(Xtest, Ytest, '-o', color='green')
        plt.plot_date(Xtest, ypred_test, '--', color='red')
        projX = Xlonger[len(Ytest):]
        projY = longer_pred[len(Ytest):]
    else:
        projX = Xlonger
        projY = longer_pred
    plt.plot_date(projX, projY, '--', color='blue')
    plt.xlim(1967, 2016+pred_len)
    plt.xticks(np.arange(1970, 2015+pred_len, 5),
               np.arange(1970, 2015+pred_len, 5).astype(str))
    plt.show()

    output_dict['comp_time_sec'] = comp_time_sec
    return output_dict


def main(argv):
    try:
        file_method = argv[0]
    except:
        file_method = None

    # get and clean data
    older_df = load_data('older')
    newer_df = load_data('newer')

    older_df, old_dropped_df = clean_nans(older_df, 'older')
    newer_df, new_dropped_df = clean_nans(newer_df, 'newer')

    # normalize values to a relative quantaty
    older_df = relative_normalize(older_df, 'older')
    newer_df = relative_normalize(newer_df, 'newer')

    df = merge_dfs(older_df, newer_df)

    # consider one case to start modeling:
    randomly_chosen_geo = '27100'
    print 'selected example city: %s' % \
        df.loc[randomly_chosen_geo].loc['OldGeoName']

    years = np.array([str(yr) for yr in range(1969, 2014)])
    values = df.loc[randomly_chosen_geo].loc[years].values
    years = years.astype(int)

    '''
    pool = mp.Pool(processes=mp.cpu_count()-1)
    results = [pool.apply_async(read_file_do_sentiment,
        args=(infile,)) for infile in onlyfiles]
    output = [p.get() for p in results]'''

    geo = randomly_chosen_geo
    ind = 'manf'
    wind = 4
    ntst = 0

    model_filename = 'results/regional_income/'\
        'fitted_model_g%s_i%s_w%d_t%d.shelf' % (geo, ind, wind, ntst)
    results_filename = 'results/regional_income/'\
        'results_g%s_i%s_w%d_t%d.shelf' % (geo, ind, wind, ntst)

    key = 'args_dict'

    try:

        if file_method != 'load':
            # signal handler for a timeout clock
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(350)

            write_args_dict = windowed_svr_fit_model(years,
                                                     values,
                                                     ntest=ntst,
                                                     window=wind)

            writeshelf = shelve.open(model_filename, 'n')
            writeshelf[key] = write_args_dict
            writeshelf.close()

            signal.alarm(0)

            time.sleep(2)

        if file_method != 'save':
            readshelf = shelve.open(model_filename, 'r')
            read_args_dict = readshelf[key]
            readshelf.close()

            results_dict = windowed_svr_get_result(read_args_dict)

            writeshelf = shelve.open(results_filename, 'n')
            writeshelf[key] = results_dict
            writeshelf.close()

    except MyTimeoutException, exc:
        print exc

    print 'done!'


if __name__ == "__main__":
    main(sys.argv[1:])
