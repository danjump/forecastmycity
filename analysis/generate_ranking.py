#!/usr/bin/python

import sys
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sklearn.linear_model import LinearRegression


def get_geo_list(industry):
    engine = create_engine("mysql+mysqldb://danielj:@localhost/forecastmycity")
    con = engine.connect()

    query = 'SELECT DISTINCT(geofips) FROM fit_data_pop WHERE '\
        'industry="%s"' % industry

    df = pd.read_sql(query, con=con.connection)

    return df


def get_one_dataset(geofips, industry, method, which_case):
    engine = create_engine("mysql+mysqldb://danielj:@localhost/forecastmycity")
    con = engine.connect()

    query = 'SELECT x, y FROM fit_data_pop WHERE '\
        'which_case="%s" and geofips="%s" and industry="%s" and method="%s"' %\
        (which_case, geofips, industry, method)

    df = pd.read_sql(query, con=con.connection)

    return df


def linear_fit(df):
    lr = LinearRegression(fit_intercept=True)

    x = df['x'].values.reshape(len(df), 1)
    y = df['y'].values
    lr.fit(x, y)

    pred = lr.predict(x)

    return pd.DataFrame({'x': df['x'].values, 'y': pred})


def plot_comparisons(history, data, data_lr, proj, proj_lr):
    data_m1 = data['y'].iloc[:5].mean()
    data_m2 = data['y'].iloc[5:].mean()
    proj_m1 = proj['y'].iloc[:5].mean()
    proj_m2 = proj['y'].iloc[5:].mean()

    plt.plot(history['x'].values, history['y'].values, '-o', color='blue')
    plt.plot(data['x'].values, data['y'].values, '-o', color='green')
    plt.plot(data_lr['x'].values, data_lr['y'].values, '--', color='green')
    plt.plot(range(2004, 2009),
             [data_m1 for i in range(5)],
             '--', color='black')
    plt.plot(range(2009, 2014),
             [data_m2 for i in range(5)],
             '--', color='black')
    plt.plot(proj['x'].values, proj['y'].values, '-o', color='red')
    plt.plot(proj_lr['x'].values, proj_lr['y'].values, '--', color='red')
    plt.plot(range(2014, 2019),
             [proj_m1 for i in range(5)],
             '--', color='black')
    plt.plot(range(2019, 2024),
             [proj_m2 for i in range(5)],
             '--', color='black')
    plt.show()


def write_rankings_to_sql(geofips, industry,
                          past10_avg, past10_slope, proj10_avg, proj10_slope,
                          past5_avg, past5_slope, proj5_avg, proj5_slope):
    fill_dict = {}
    fill_dict['geofips'] = [geofips]
    fill_dict['industry'] = [industry]
    fill_dict['past10_avg'] = [past10_avg]
    fill_dict['past10_slope'] = [past10_slope]
    fill_dict['proj10_avg'] = [proj10_avg]
    fill_dict['proj10_slope'] = [proj10_slope]
    fill_dict['past5_avg'] = [past5_avg]
    fill_dict['past5_slope'] = [past5_slope]
    fill_dict['proj5_avg'] = [proj5_avg]
    fill_dict['proj5_slope'] = [proj5_slope]

    df = pd.DataFrame(fill_dict)
    try:
        engine = create_engine(
            "mysql+mysqldb://danielj:@localhost/forecastmycity")
    except:
        print 'Error connecting to db:', sys.exc_info()[0]
        return 0
    try:
        df.to_sql(con=engine, name='ranking_scores_pop', index=False,
                  if_exists='append', flavor='mysql')
    except OperationalError as err:
        print 'Error writing to db:', sys.exc_info()[0]
        print err.message
        return 0
    return 1


def get_scores(data_df, proj_df, years, do_plots):
    past = data_df.iloc[-years:]
    past_lr = linear_fit(past)
    proj = proj_df.iloc[:years]
    proj_lr = linear_fit(proj)

    past_midyear = 2013-years//2
    past_avg = past_lr[past_lr['x'] == past_midyear]['y'].values
    past_slope = past_avg - \
        past_lr[past_lr['x'] == past_midyear - 1]['y'].values
    proj_midyear = 2013+years//2
    proj_avg = proj_lr[proj_lr['x'] == proj_midyear]['y'].values
    proj_slope = proj_avg - \
        proj_lr[proj_lr['x'] == proj_midyear - 1]['y'].values

    if do_plots:
        plot_comparisons(data_df.iloc[:-years], past, past_lr, proj, proj_lr)

    return past_avg[0], past_slope[0], proj_avg[0], proj_slope[0]


def main(argv):
    industries = ['manf', 'rettrd', 'gov']
    for ind in industries:
        geo_df = get_geo_list(ind)
        entries = len(geo_df)

        for geo, count in zip(geo_df['geofips'], range(entries)):
            sys.stdout.write('\r%s Progress: %d/%d' % (ind, count+1, entries))
            sys.stdout.flush()

            data = get_one_dataset(geo, ind, 'AR_linreg', 'full_data')
            proj = get_one_dataset(geo, ind, 'AR_linreg', 'full_proj')

            do_plots = False
            past10_avg, past10_slope, proj10_avg, proj10_slope = \
                get_scores(data, proj, 10, do_plots=do_plots)
            past5_avg, past5_slope, proj5_avg, proj5_slope = \
                get_scores(data, proj, 5, do_plots=do_plots)

            write_rankings_to_sql(geo, ind,
                                  past10_avg, past10_slope,
                                  proj10_avg, proj10_slope,
                                  past5_avg, past5_slope,
                                  proj5_avg, proj5_slope)
        print ''


if __name__ == "__main__":
    main(sys.argv[1:])
