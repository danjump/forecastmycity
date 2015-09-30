#!/usr/bin/python

import sys
import signal
import time
import shelve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
# import sklearn.svm as svm
# from sklearn.grid_search import GridSearchCV
import sklearn.linear_model as linear_model
import sklearn.metrics as metric
from sklearn.preprocessing import StandardScaler
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
        # industry_code = 500  # manf
        industry_code = 700  # rettrd
        # industry_code = 2000  # gov
        year_cols = [str(yr) for yr in range(2001, 2014)]
    elif which == 'older':
        infile = '../data/BEA-RegionalIncomeByIndustry/CA5_1969_2000_MSA.csv'
        # industry_code = 400  # manf
        industry_code = 620  # rettrd
        # industry_code = 900  # gov
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


def windowed_svr_fit_model(X, Y, geo, ind, ntest=10, window=5):
    ''' creates and fits an SVR model to data.
    accepts np.arrays X and Y'''
    start_time = datetime.datetime.now()

    def generate_windows(arr):
        numPts = len(arr)

        i = 0
        j = window
        x = []
        y = []

        # each row of x is a window of time points
        while j < numPts:
            x.append(arr[i:j])
            i += 1
            j += 1

        # each entry in y is the time point after the corresponding x window
        j = window
        while j < numPts:
            y.append(arr[j])
            j += 1
        return x, y

    X_full = X
    Y_full = Y

    X_train = X[0:-ntest]
    Y_train = Y[0:-ntest]

    X_test = X[-ntest:]
    Y_test = Y[-ntest:]

    x_full, y_full = generate_windows(Y_full)
    x_train, y_train = generate_windows(Y_train)

    # scale the data for fitting in the model
    trans = StandardScaler()
    Y = trans.fit_transform(Y)

    # windowed svr gridsearch
    '''
    c_set = []
    for ex in range(-50, 50):
        for fac in range(1, 10):
            c_set.append(fac*(2**ex))
    parameters = {'C': c_set}

    svr_test = svm.LinearSVR(random_state=873487)
    svr_full = svm.LinearSVR(random_state=873487)

    # do gridsearch fit of model
    grid_test = GridSearchCV(svr_test, param_grid=parameters, cv=5, n_jobs=1)
    grid_full = GridSearchCV(svr_full, param_grid=parameters, cv=5, n_jobs=1)

    model_train = grid_test.fit(x_train, y_train)
    model_full = grid_full.fit(x_full, y_full)
    '''
    linear_train = linear_model.LinearRegression(fit_intercept=False)
    linear_full = linear_model.LinearRegression(fit_intercept=False)
    model_train = linear_train.fit(x_train, y_train)
    model_full = linear_full.fit(x_full, y_full)

    end_time = datetime.datetime.now()
    delta_time = end_time - start_time
    comp_time_sec = delta_time.total_seconds()  # seconds
    print 'comp_time:', comp_time_sec

    ret_dict = {'model_train': model_train,
                'model_full': model_full,
                'window': window,
                'ntest': ntest,
                'geo': geo,
                'ind': ind,
                'transform': trans,
                'x_train': x_train,
                'y_train': y_train,
                'x_full': x_full,
                'y_full': y_full,
                'X_train': X_train,
                'Y_train': Y_train,
                'X_full': X_full,
                'Y_full': Y_full,
                'X_test': X_test,
                'Y_test': Y_test,
                'comp_time_sec': comp_time_sec}

    return ret_dict


def windowed_svr_get_result(args_dict):
    model_train = args_dict['model_train']
    model_full = args_dict['model_full']
    window = args_dict['window']
    ntest = args_dict['ntest']
    geo = args_dict['geo']
    ind = args_dict['ind']
    trans = args_dict['transform']
    x_train = args_dict['x_train']
    y_train = args_dict['y_train']
    x_full = args_dict['x_full']
    y_full = args_dict['y_full']
    X_train = args_dict['X_train']
    Y_train = args_dict['Y_train']
    X_full = args_dict['X_full']
    Y_full = args_dict['Y_full']
    X_test = args_dict['X_test']
    Y_test = args_dict['Y_test']
    comp_time_sec = args_dict['comp_time_sec']

    output_dict = {}

    def forecast_iteration(x, n_pred, model):
        if n_pred > 1:
            y = model.predict(x)
            y_curr = [y]
            next_x = np.append(x[1:], y_curr)
            y_next_deeper = forecast_iteration(next_x, n_pred-1, model)
            y_previous_shallower = np.append(y_curr, y_next_deeper)
            return y_previous_shallower
        else:
            y = model.predict(x)
            return [y]

    def run_metrics(y, ypred):
        score_r2 = metric.r2_score(y, ypred)
        score_expvar = metric.explained_variance_score(y, ypred)
        score_meansqrerr = metric.mean_squared_error(y, ypred)
        score_meanabserr = metric.mean_absolute_error(y, ypred)
        metric_dict = {'r2': score_r2,
                       'ev': score_expvar,
                       'mse': score_meansqrerr,
                       'mae': score_meanabserr}
        return metric_dict

    ypred_train = model_train.predict(x_train)
    output_dict['train_metrics'] = run_metrics(y_train, ypred_train)

    ypred_full = model_full.predict(x_full)
    output_dict['full_metrics'] = run_metrics(y_full, ypred_full)

    ypred_test = forecast_iteration(Y_train[-window:], len(Y_test), model_train)
    output_dict['test_metrics'] = run_metrics(Y_test, ypred_test)
    print ' metrics train: ', output_dict['train_metrics']
    print ' metrics test: ', output_dict['test_metrics']

    pred_len = 10
    yproj_train = forecast_iteration(Y_train[-window:],
                                     pred_len+window, model_train)[window:]
    Xproj_train = [yr for yr in range(
        X_test[-1] + 1, X_test[-1] + 1 + pred_len)]
    yproj_full = forecast_iteration(Y_full[-window:], pred_len, model_full)
    Xproj_full = [yr for yr in range(
        X_full[-1] + 1, X_full[-1] + 1 + pred_len)]

    y_train = trans.inverse_transform(y_train)
    y_full = trans.inverse_transform(y_full)
    ypred_train = trans.inverse_transform(ypred_train)
    ypred_full = trans.inverse_transform(ypred_full)
    ypred_test = trans.inverse_transform(ypred_test)
    yproj_train = trans.inverse_transform(yproj_train)
    yproj_full = trans.inverse_transform(yproj_full)
    Y_full = trans.inverse_transform(Y_full)
    Y_train = trans.inverse_transform(Y_train)
    Y_test = trans.inverse_transform(Y_test)

    output_dict['scaled_train_metrics'] = run_metrics(y_train, ypred_train)
    output_dict['scaled_full_metrics'] = run_metrics(y_full, ypred_full)
    output_dict['scaled_test_metrics'] = run_metrics(Y_test, ypred_test)
    print ' scaled metrics train: ', output_dict['scaled_train_metrics']
    print ' scaled metrics test: ', output_dict['scaled_test_metrics']

    output_dict['ypred_train'] = ypred_train
    output_dict['ypred_full'] = ypred_train
    output_dict['ypred_test'] = ypred_test
    output_dict['Xproj_train'] = Xproj_train
    output_dict['Xproj_full'] = Xproj_full
    output_dict['yproj_train'] = yproj_train
    output_dict['yproj_full'] = yproj_full
    output_dict['X_train'] = X_train
    output_dict['Y_train'] = Y_train
    output_dict['X_full'] = X_full
    output_dict['Y_full'] = Y_full
    output_dict['X_test'] = X_test
    output_dict['Y_test'] = Y_test
    output_dict['geofips'] = geo
    output_dict['industry'] = ind
    output_dict['window'] = window
    output_dict['ntest'] = ntest

    # plt.plot_date(X_full, Y_full, '-o', color='green')
    # plt.plot_date(X_full[window:], ypred_full, '--', color='red')
    plt.plot_date(X_train, Y_train, '-o', color='green')
    plt.plot_date(X_train[window:], ypred_train, '--', color='red')
    plt.plot_date(X_test, Y_test, '-o', color='green')
    plt.plot_date(X_test, ypred_test, '--', color='red')
    plt.plot_date(Xproj_train, yproj_train, '--', color='blue')
    plt.plot_date(Xproj_full, yproj_full, '--', color='blue')
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

    for geofips, count in zip(df.index, range(0, len(df))):
        geo = geofips
        ind = 'test'
        wind = 7
        ntst = 10

        # consider one case to start modeling:
        # randomly_chosen_geo = '27100'
        print '%dCity %s: %s\n' % \
            (count, geo, df.loc[geo].loc['OldGeoName'])

        years = np.array([str(yr) for yr in range(1969, 2014)])
        values = df.loc[geo].loc[years].values
        years = years.astype(int)

        '''
        pool = mp.Pool(processes=mp.cpu_count()-1)
        results = [pool.apply_async(read_file_do_sentiment,
            args=(infile,)) for infile in onlyfiles]
        output = [p.get() for p in results]'''

        model_filename = 'results/regional_income/'\
            'fitted_model_g%s_i%s_w%d_t%d.shelf' % (geo, ind, wind, ntst)
        results_filename = 'results/regional_income/'\
            'results_g%s_i%s_w%d_t%d.shelf' % (geo, ind, wind, ntst)

        key = 'args_dict'

        try:

            if file_method != 'load':
                # signal handler for a timeout clock
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(425)

                write_args_dict = windowed_svr_fit_model(years,
                                                         values,
                                                         geo,
                                                         ind,
                                                         ntest=ntst,
                                                         window=wind)

                print '%dSaving model: %s' % (count, model_filename)
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

                print '%dSaving results: %s' % (count, results_filename)
                writeshelf = shelve.open(results_filename, 'n')
                writeshelf[key] = results_dict
                writeshelf.close()

        except MyTimeoutException, exc:
            print exc
            print '%d---------GEOFIPS %s FAILED!!!----------\n' % (count, geo)

    print 'done!'
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
