#!/usr/bin/python

import sys
import pandas as pd
from sqlalchemy import create_engine


def get_attributes(industry):
    engine = create_engine("mysql+mysqldb://danielj:@localhost/ecotest")
    con = engine.connect()

    query = 'SELECT DISTINCT(geofips) FROM fit_data WHERE '\
        'industry="%s"' % industry

    df = pd.read_sql(query, con=con.connection)

    return df


def get_one_dataset(geofips, industry, method, which_case):
    engine = create_engine("mysql+mysqldb://danielj:@localhost/ecotest")
    con = engine.connect()

    query = 'SELECT x, y FROM fit_data WHERE '\
        'which_case="%s" and geofips="%s" and industry="%s" and method="%s"' %\
        (which_case, geofips, industry, method)

    df = pd.read_sql(query, con=con.connection)

    return df


def main(argv):
    geo_df = get_attributes('manf')
    print geo_df


if __name__ == "__main__":
    main(sys.argv[1:])
