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

    query = 'SELECT DISTINCT(NewGeoName) FROM info_pop WHERE '\
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

    if industry == 'manf':
        lim = 11
    else:
        lim = 10

    query = 'SELECT *, '\
        '(proj5_avg + proj5_slope) as proj5_sum FROM ranking_scores_pop WHERE '\
        'industry="%s"' % (industry) +\
        'ORDER BY proj5_sum DESC LIMIT %d' % lim

    df = pd.read_sql(query, con=con.connection)
    if industry == 'manf':
        df.drop(df.index[1], inplace=True)

    df.loc[:, 'proj5_avg'] *= 100
    df.loc[:, 'past5_avg'] *= 100

    return df


def get_datapoints_from_sql(case, industry, geofips):
    engine = create_engine("mysql+mysqldb://root:@localhost/forecastmycity")
    con = engine.connect()

    query = 'SELECT x, y FROM fit_data_pop WHERE '\
        'which_case="%s" and geofips="%s" and industry="%s" and method="%s"' %\
        (case, geofips, industry, 'AR_linreg')

    df = pd.read_sql(query, con=con.connection)
    df.loc[:, 'y'] *= 100

    return df.values


@fmc_app.route('/output')
def cities_output():
    ind_name = request.args.get('Industry')

    if ind_name == 'Manufacturing':
        industry = 'manf'
    elif ind_name == 'Retail Trade':
        industry = 'rettrd'
    elif ind_name == 'Government':
        industry = 'gov'
    else:
        ind_name = 'Retail Trade'
        industry = 'rettrd'
    ranking_df = get_rankings_from_sql(industry)

    geo_list = ranking_df['geofips'].values
    geo_names = []
    result_list = []
    for geo in geo_list:
        geo_names.append(get_geo_name(geo))
        past5 = ranking_df[ranking_df['geofips'] == geo]['past5_avg'].values[0]
        proj5 = ranking_df[ranking_df['geofips'] == geo]['proj5_avg'].values[0]
        result_list.append({'name': geo_names[-1],
                            'past': '%.1f %% above avg.' % past5,
                            'proj': '%.1f %% above avg.' % proj5})

    data = {}
    proj = {}
    for geo in geo_list[:5]:
        data[geo] = get_datapoints_from_sql('full_data', industry, geo)
        proj[geo] = get_datapoints_from_sql('full_proj', industry, geo)

    # colorblind friendly palette:
    # "#000000", "#E69F00", "#56B4E9", "#009E73",
    # "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
    custom_style = pygal.style.Style(
        label_font_size=20, major_label_font_size=20,
        colors=('#000000', '#000000',
                '#CC79A7', '#CC79A7',
                '#56B4E9', '#56B4E9',
                '#009E73', '#009E73',
                '#F0E442', '#F0E442'))

    chart = pygal.XY(disable_xml_declaration=True, width=800, height=350,
                     style=custom_style, truncate_legend=-1,
                     legend_at_right=True,
                     x_label_rotation=45, y_title='% Above Average')
    chart.title = '%s Earnings Compared to Average' % ind_name
    chart.x_labels = map(int, range(1965, 2025, 5))
    for i, geo in enumerate(geo_list[:5]):
        name = geo_names[i]
        chart.add(re.match('[^-^,]+', name).group(), data[geo], show_dots=False,
                  stroke_style={'width': 3,
                                'linecap': 'round', 'linejoin': 'round'})
        print name
        chart.add(name[-3:-1]+' - #%d' % (i+1),
                  proj[geo], show_dots=False,
                  stroke_style={'width': 3, 'dasharray': '3, 6',
                                'linecap': 'round', 'linejoin': 'round'})

    return render_template('output.html', industry=ind_name,
                           result_list=result_list, chart=chart)
