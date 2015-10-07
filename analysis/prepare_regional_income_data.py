#!/usr/bin/python

import sys
import pandas as pd
import numpy as np
import re
# import sklearn.svm as svm
# from sklearn.grid_search import GridSearchCV
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
# import datetime


def load_data(which):
    if which == 'newer':
        infile = '../data/BEA-RegionalIncomeByIndustry/CA5N_2001_2013_MSA.csv'
    elif which == 'older':
        infile = '../data/BEA-RegionalIncomeByIndustry/CA5_1969_2000_MSA.csv'

    df = pd.read_csv(infile, low_memory=False)

    return df


def select_industry(df, industry, which):
    if which == 'newer':
        if industry == 'manf':
            industry_code = 500  # manf
        elif industry == 'rettrd':
            industry_code = 700  # rettrd
        elif industry == 'gov':
            industry_code = 2000  # gov
    elif which == 'older':
        if industry == 'manf':
            industry_code = 400  # manf
        elif industry == 'rettrd':
            industry_code = 620  # rettrd
        elif industry == 'gov':
            industry_code = 900  # gov

    # select a specific industry
    ind_df = df[df['LineCode'] == industry_code]

    return ind_df


def read_employment_from_sql(code, geo, dbname='test'):
    engine = create_engine("mysql+mysqldb://danielj:@localhost/%s" % dbname)
    con = engine.connect()

    query = 'SELECT DataValue, TimePeriod FROM employment_dataset WHERE '\
        'Code like "%%%s%%" and GeoFIPS = "%s"' % (code, geo)

    df = pd.read_sql(query, con=con.connection)

    return df


def scale_by_employment(ear_df):
    nan_count = 0
    for i in range(len(ear_df)):
        code = re.sub('CA5', 'CA25', ear_df.iloc[i]['Table']) + '-' + \
            '%d' % ear_df.iloc[i]['LineCode']
        geo = ear_df.iloc[i]['GeoFIPS']

        emp_df = read_employment_from_sql(code, geo)

        def clean_data_entry(x):
            try:
                x = re.sub(',', '', x)
                m = re.match("\d+", x)
                # i.e. - if the string starts with
                if m:
                    return float(m.group())
                else:
                    return np.nan
            except:
                return x

        was_nan = False
        for data, year in emp_df.values:
            emp = clean_data_entry(data)

            if not np.isnan(emp):
                ear_df.loc[ear_df.index[i], year] = ear_df.iloc[i][year] / emp
            else:
                was_nan = True
                ear_df.loc[ear_df.index[i], year] = np.nan

        if was_nan:
            nan_count += 1
            print geo, code

    print nan_count
    return ear_df


def clean_nans(df, which, str_check=True):
    if which == 'newer':
        year_cols = [str(yr) for yr in range(2001, 2014)]
    elif which == 'older':
        year_cols = [str(yr) for yr in range(1969, 2001)]

    if str_check:
        # utility function to remove characters from number strings
        def clean_data_entry(x):
            try:
                m = re.match("\d+", x)
                # i.e. - if the string starts with
                if m:
                    return float(m.group())
                else:
                    return np.nan
            except:
                return x

        df[year_cols] = df[year_cols].applymap(clean_data_entry)

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

    return df, means_df


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


def main(argv):
    try:
        if argv[0] == 'test':
            dbname = 'test'
        else:
            dbname = 'forecastmycity'
    except:
        dbname = 'forecastmycity'

    industries = ['manf', 'rettrd', 'gov']

    for industry in industries:
        print 'Starting %s fits:' % industry

        attributes = {'industry': industry,
                      'wind': 7,
                      'ntst': 10}
        # get and clean data
        older_df = load_data('older')
        newer_df = load_data('newer')

        older_df, old_dropped_df = clean_nans(older_df, 'older')
        newer_df, new_dropped_df = clean_nans(newer_df, 'newer')

        print older_df.head()

        older_df = select_industry(older_df, attributes['industry'], 'older')
        newer_df = select_industry(newer_df, attributes['industry'], 'newer')

        print older_df.head()

        older_df = scale_by_employment(older_df)
        newer_df = scale_by_employment(newer_df)

        print older_df.head()
        print len(older_df)

        older_df, old_dropped_df = clean_nans(older_df, 'older', str_check=True)
        newer_df, new_dropped_df = clean_nans(newer_df, 'newer', str_check=True)

        print len(older_df)

        # normalize values to a relative quantaty
        older_df, older_mean_df = relative_normalize(older_df, 'older')
        newer_df, newer_mean_df = relative_normalize(newer_df, 'newer')

        print len(older_df)
        print older_df.head()

        df = merge_dfs(older_df, newer_df)

        write_to_sql(df.reset_index(level=0), 'dataset_pop', dbname, 'append')


if __name__ == "__main__":
    main(sys.argv[1:])
