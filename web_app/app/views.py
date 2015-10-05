import pandas as pd
# import numpy as np
import pygal
import re
from flask import render_template, request
from app import fmc_app
from sqlalchemy import create_engine


@fmc_app.route('/')
@fmc_app.route('/input')
def cities_input():
    return render_template('input.html')


def get_geo_name(geofips):
    engine = create_engine("mysql+mysqldb://root:@localhost/forecastmycity")
    con = engine.connect()

    query = 'SELECT DISTINCT(NewGeoName) FROM info WHERE '\
        'GeoFIPS="%s"' % (geofips)

    df = pd.read_sql(query, con=con.connection)
    name = df.values[0][0]
    m = re.match("[^(]+", name)
    name = m.group()
    print name

    return name


def get_rankings_from_sql(industry):
    engine = create_engine("mysql+mysqldb://root:@localhost/forecastmycity")
    con = engine.connect()

    query = 'SELECT *, '\
        '(proj5_avg + proj5_slope) as proj5_sum FROM ranking_scores WHERE '\
        'industry="%s"' % (industry) +\
        'ORDER BY proj5_sum DESC LIMIT 10'

    df = pd.read_sql(query, con=con.connection)

    return df


def get_datapoints_from_sql(case, industry, geofips):
    engine = create_engine("mysql+mysqldb://root:@localhost/forecastmycity")
    con = engine.connect()

    query = 'SELECT x, y FROM fit_data WHERE '\
        'which_case="%s" and geofips="%s" and industry="%s" and method="%s"' %\
        (case, geofips, industry, 'AR_linreg')

    df = pd.read_sql(query, con=con.connection)

    return df.values


@fmc_app.route('/output')
def cities_output():
    industry = request.args.get('Industry')

    industry = 'manf'
    ranking_df = get_rankings_from_sql(industry)

    geo_list = ranking_df['geofips'].values
    geo_names = []
    result_list = []
    for geo in geo_list:
        geo_names.append(get_geo_name(geo))
        past = ranking_df[ranking_df['geofips'] == geo]['past5_avg'].values[0]
        proj = ranking_df[ranking_df['geofips'] == geo]['proj5_avg'].values[0]
        result_list.append({'name': geo_names[-1],
                            'past': past,
                            'proj': proj})

    data = {}
    proj = {}
    for geo in geo_list[:5]:
        data[geo] = get_datapoints_from_sql('full_data', industry, geo)
        proj[geo] = get_datapoints_from_sql('full_proj', industry, geo)

    chart = pygal.XY(disable_xml_declaration=True, width=800, height=350)
    chart.title = '%s Earnings Compared to Average' % ind_name
    chart.x_labels = map(int, range(1965, 2025, 5))
    for geo in geo_list[:5]:
        chart.add('Data', data[geo])
        chart.add('Projection', proj[geo])

    return render_template('output.html', industry=industry,
                           result_list=result_list, chart=chart)
