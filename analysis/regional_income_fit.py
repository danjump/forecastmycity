#!/usr/bin/python

import sys
import signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import sklearn.svm as svm
# from sklearn.grid_search import GridSearchCV
import sklearn.linear_model as linear_model
import sklearn.metrics as metric
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
# import datetime
# import multiprocessing as mp


class MyTimeoutException(Exception):
    pass


def handler(signum, frame):
    print "Fit ran out of time!"
    raise MyTimeoutException("fit timeout")


def get_data_ranges(X, Y, which_case, window, ntest):
    def generate_windows(arr):
        numPts = len(arr)
        window = 7

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

    results = {}
    if which_case == 'full':
        results['X'] = X
        results['Y'] = Y
        results['x'], results['y'] = generate_windows(Y)

    if which_case == 'test':
        results['X'] = X[0:-ntest]
        results['Y'] = Y[0:-ntest]

        results['X_test'] = X[-ntest:]
        results['Y_test'] = Y[-ntest:]

        results['x'], results['y'] = generate_windows(results['Y'])

    results['X_proj'] = range(X[-1]+1, X[-1]+1+10)

    return results


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


def append_coords(results, append_list):
    for append_item in append_list:
        X = append_item[0]
        Y = append_item[1]
        which_case = append_item[2]

        for xval, yval in zip(X, Y):
            results['which_case'].append(which_case)
            results['x'].append(xval)
            results['y'].append(yval)

    return results


def dofit_linreg(X, Y, which_case='both'):
    '''accepts np.arrays X and Y'''
    ntest = 10
    pred_len = 10

    if which_case == 'both':
        # start_time = datetime.datetime.now()

        full_data = dofit_linreg(X, Y, 'full')
        test_data = dofit_linreg(X, Y, 'test')

        results = {}
        results['which_case'] = []
        results['x'] = []
        results['y'] = []

        results = append_coords(
            results,
            [(full_data['x'], full_data['y'], 'full_data'),
             (full_data['x'], full_data['ypred'], 'full_pred'),
             (full_data['xproj'], full_data['yproj'], 'full_proj'),
             (test_data['x'], test_data['x'], 'train_data'),
             (test_data['x'], test_data['ypred'], 'train_pred'),
             (test_data['xtest'], test_data['ytest'], 'test_data'),
             (test_data['xtest'], test_data['ypred_test'], 'test_pred'),
             (test_data['xproj'], test_data['yproj'], 'train_proj')])

        result_df = pd.DataFrame(results)

        metrics = {}
        metrics['which_range'] = []
        metrics['mse'] = []
        metrics['ev'] = []
        metrics['mae'] = []
        metrics['r2'] = []

        metrics['which_range'].append('full')
        for key in full_data['metrics']:
            metrics[key].append(full_data['metrics'][key])
        metrics['which_range'].append('train')
        for key in test_data['metrics']:
            metrics[key].append(test_data['metrics'][key])
        metrics['which_range'].append('test')
        for key in test_data['metrics']:
            metrics[key].append(test_data['metrics_test'][key])

        metric_df = pd.DataFrame(metrics)

        '''
        end_time = datetime.datetime.now()
        delta_time = end_time - start_time
        comp_time_sec = delta_time.total_seconds()  # seconds
        print 'comp_time:', comp_time_sec
        '''

        '''
        plt.plot_date(test_data['x'], test_data['y'], '-o', color='green')
        plt.plot_date(
            test_data['x'], test_data['ypred'], '--', color='red')

        plt.plot_date(
            test_data['xtest'], test_data['ytest'], '-o', color='green')
        plt.plot_date(
            test_data['xtest'], test_data['ypred_test'], '--', color='red')

        plt.plot_date(
            full_data['x'], full_data['ypred'], '--', color='blue')

        plt.plot_date(
            test_data['xproj'], test_data['yproj'], '--', color='red')
        plt.plot_date(
            full_data['xproj'], full_data['yproj'], '--', color='blue')
        plt.xlim(1967, 2016+pred_len)
        plt.xticks(np.arange(1970, 2015+pred_len, 5),
                   np.arange(1970, 2015+pred_len, 5).astype(str))
        plt.show()
        '''

        return result_df, metric_df
    else:
        # generate time-windowed data points as well as split test region
        # 'data' is a dictionary of various x and y arrays
        data = {}
        if which_case == 'test':
            data['x'] = X[:-ntest]
            data['y'] = Y[:-ntest]
            data['xtest'] = X[-ntest:]
            data['ytest'] = Y[-ntest:]
        else:
            data['x'] = X[:]
            data['y'] = Y[:]

        linear = linear_model.LinearRegression(fit_intercept=True)
        model = linear.fit(data['x'].reshape(len(data['x']), 1), data['y'])

        # get prediction for data time period
        data['ypred'] = model.predict(data['x'].reshape(len(data['x']), 1))

        if which_case == 'test':
            data['ypred_test'] = model.predict(
                data['xtest'].reshape(len(data['xtest']), 1))

        # get prediction for forecast time period
        data['xproj'] = np.array(range(X[-1]+1, X[-1]+1+pred_len))
        data['yproj'] = model.predict(data['xproj'].reshape(pred_len, 1))

        # get fit scores
        data['metrics'] = run_metrics(data['y'], data['ypred'])

        if which_case == 'test':
            data['metrics_test'] = run_metrics(data['ytest'],
                                               data['ypred_test'])

        return data


def dofit_AR_linreg(X, Y, which_case='both'):
    '''accepts np.arrays X and Y'''
    window = 7
    ntest = 10
    pred_len = 10

    if which_case == 'both':
        # start_time = datetime.datetime.now()

        full_data = dofit_AR_linreg(X, Y, 'full')
        test_data = dofit_AR_linreg(X, Y, 'test')

        results = {}
        results['which_case'] = []
        results['x'] = []
        results['y'] = []

        results = append_coords(
            results,
            [(full_data['X'], full_data['Y'], 'full_data'),
             (full_data['X'][window:], full_data['ypred'], 'full_pred'),
             (full_data['X_proj'], full_data['yproj'], 'full_proj'),
             (test_data['X'], test_data['Y'], 'train_data'),
             (test_data['X'][window:], test_data['ypred'], 'train_pred'),
             (test_data['X_test'], test_data['Y_test'], 'test_data'),
             (test_data['X_test'], test_data['ypred_test'], 'test_pred'),
             (test_data['X_proj'], test_data['yproj'], 'train_proj')])

        result_df = pd.DataFrame(results)

        metrics = {}
        metrics['which_range'] = []
        metrics['mse'] = []
        metrics['ev'] = []
        metrics['mae'] = []
        metrics['r2'] = []

        metrics['which_range'].append('full')
        for key in full_data['metrics']:
            metrics[key].append(full_data['metrics'][key])
        metrics['which_range'].append('train')
        for key in test_data['metrics']:
            metrics[key].append(test_data['metrics'][key])
        metrics['which_range'].append('test')
        for key in test_data['metrics']:
            metrics[key].append(test_data['metrics_test'][key])

        metric_df = pd.DataFrame(metrics)

        '''
        end_time = datetime.datetime.now()
        delta_time = end_time - start_time
        comp_time_sec = delta_time.total_seconds()  # seconds
        print 'comp_time:', comp_time_sec
        '''

        plt.plot_date(test_data['X'], test_data['Y'], '-o', color='green')
        # plt.plot_date(
        #    test_data['X'][window:], test_data['ypred'], '--', color='red')

        plt.plot_date(
            test_data['X_test'], test_data['Y_test'], '-o', color='green')
        # plt.plot_date(
        #    test_data['X_test'], test_data['ypred_test'], '--', color='red')

        plt.plot_date(
            full_data['X'][window:], full_data['ypred'], '--', color='blue')

        # plt.plot_date(
        #    test_data['X_proj'], test_data['yproj'], '--', color='red')
        plt.plot_date(
            full_data['X_proj'], full_data['yproj'], '--', color='red')
        plt.xlim(1967, 2016+pred_len)
        plt.xticks(np.arange(1970, 2015+pred_len, 5),
                   np.arange(1970, 2015+pred_len, 5).astype(str))
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(24)
            tick.label.set_rotation('vertical')
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(24)
        plt.subplots_adjust(bottom=.2)
        plt.show()

        return result_df, metric_df
    else:
        # scale the data for fitting in the model
        trans = StandardScaler()
        Y = trans.fit_transform(Y)

        # generate time-windowed data points as well as split test region
        # 'data' is a dictionary of various x and y arrays
        data = get_data_ranges(X, Y, which_case, window, ntest)

        linear = linear_model.LinearRegression(fit_intercept=False)
        model = linear.fit(data['x'], data['y'])

        # get prediction for data time period
        data['ypred'] = model.predict(data['x'])

        if which_case == 'test':
            data['ypred_test'] = forecast_iteration(
                data['Y'][-window:], ntest, model)

        # get prediction for forecast time period
        if which_case == 'test':
            length = ntest + pred_len
        else:
            length = pred_len
        data['yproj'] = forecast_iteration(
            data['Y'][-window:], length, model)[-pred_len:]

        # inverse the scaling transformations
        data['y'] = trans.inverse_transform(data['y'])
        data['ypred'] = trans.inverse_transform(data['ypred'])
        data['yproj'] = trans.inverse_transform(data['yproj'])
        data['Y'] = trans.inverse_transform(data['Y'])

        # get fit scores
        data['metrics'] = run_metrics(data['y'], data['ypred'])

        if which_case == 'test':
            data['ypred_test'] = trans.inverse_transform(data['ypred_test'])
            data['Y_test'] = trans.inverse_transform(data['Y_test'])
            data['metrics_test'] = run_metrics(data['Y_test'],
                                               data['ypred_test'])

        return data


def do_one_fit(X, Y):
    try:
        # signal handler for a timeout clock
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(425)

        AR_df, AR_met_df = dofit_AR_linreg(X.copy(), Y.copy())

        AR_df['method'] = ['AR_linreg' for i in range(0, len(AR_df))]
        AR_met_df['method'] = ['AR_linreg' for i in range(0, len(AR_met_df))]

        lr_df, lr_met_df = dofit_linreg(X.copy(), Y.copy())

        lr_df['method'] = ['linreg' for i in range(0, len(lr_df))]
        lr_met_df['method'] = ['linreg' for i in range(0, len(lr_met_df))]

        df = pd.concat([AR_df, lr_df], ignore_index=True)
        met_df = pd.concat([AR_met_df, lr_met_df], ignore_index=True)

        signal.alarm(0)
    except MyTimeoutException, exc:
        df = None
        print exc
        print '----------FIT FAILED!!!----------\n'

    return df, met_df


def do_all_fits(input_df, attributes):
    data_df = None
    metric_df = None
    length = len(input_df)
    for geofips, count in zip(input_df.index, range(1, length+1)):
        sys.stdout.write('\rProgress: %d/%d' % (count, length))
        sys.stdout.flush()
        geo = geofips

        years = np.array([str(yr) for yr in range(1969, 2014)])
        values = input_df.loc[geo].loc[years].values
        years = years.astype(int)

        tmp_df, tmp_met_df = do_one_fit(years, values)

        tmp_df['geofips'] = \
            [geofips for i in range(0, len(tmp_df))]
        tmp_df['industry'] = \
            [attributes['industry'] for i in range(0, len(tmp_df))]

        tmp_met_df['geofips'] = \
            [geofips for i in range(0, len(tmp_met_df))]
        tmp_met_df['industry'] = \
            [attributes['industry'] for i in range(0, len(tmp_met_df))]

        if data_df is None:
            data_df = tmp_df
            metric_df = tmp_met_df
        else:
            data_df = pd.concat([data_df, tmp_df], ignore_index=True)
            metric_df = pd.concat([metric_df, tmp_met_df], ignore_index=True)

    print ''
    return data_df, metric_df


def write_to_sql(df, tname, dbname, exist_cond):
    try:
        # print 'Connecting to database...'
        engine = create_engine(
            'mysql+mysqldb://danielj:@localhost/%s' % dbname)
    except:
        print 'Error connecting to db:', sys.exc_info()[0]
        return 0
    try:
        # print 'Writing to database table:%s'%tname
        df.to_sql(con=engine, name=tname, index=False,
                  if_exists=exist_cond, flavor='mysql')
    except OperationalError as err:
        print 'Error writing to db:', sys.exc_info()[0]
        print err.message
        return 0
    return 1


def read_from_sql(tname, dbname, industry):
    engine = create_engine("mysql+mysqldb://danielj:@localhost/%s" % dbname)
    con = engine.connect()

    if industry == 'manf':
        ind_name = 'Manufacturing'
    if industry == 'rettrd':
        ind_name = 'Retail trade'
    if industry == 'gov':
        ind_name = 'Government and government enterprises'

    query = 'SELECT * FROM dataset WHERE '\
        'NewDescription like "%%%s%%"' % ind_name

    df = pd.read_sql(query, con=con.connection)

    return df


def main(argv):
    try:
        if argv[0] == 'test':
            dbname = 'test'
        else:
            dbname = 'forecastmycity'
    except:
        dbname = 'forecastmycity'

    industries = ['manf', 'rettrd', 'gov']
    all_info_df = None
    all_data_df = None
    all_metric_df = None

    for industry in industries:
        print 'Starting %s fits:' % industry

        attributes = {'industry': industry,
                      'wind': 7,
                      'ntst': 10}

        test_df = read_from_sql('dataset', dbname, industry)

        info_df = test_df.drop([str(yr) for yr in range(1969, 2014)], axis=1)
        info_df.reset_index(level=0, inplace=True)

        data_df, metric_df = do_all_fits(test_df, attributes)

        if all_data_df is None:
            all_info_df = info_df
            all_data_df = data_df
            all_metric_df = metric_df
        else:
            all_info_df = pd.concat([all_info_df, info_df],
                                    ignore_index=True)
            all_data_df = pd.concat([all_data_df, data_df],
                                    ignore_index=True)
            all_metric_df = pd.concat([all_metric_df, metric_df],
                                      ignore_index=True)

    print len(all_data_df), all_data_df.head()
    write_to_sql(all_info_df, 'info', dbname, 'replace')
    write_to_sql(all_data_df, 'fit_data', dbname, 'replace')
    write_to_sql(all_metric_df, 'fit_metrics', dbname, 'replace')

    print 'done!'
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
