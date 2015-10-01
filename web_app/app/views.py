import MySQLdb as mdb
import pandas as pd
# import numpy as np
import pygal
from flask import render_template, request
from app import app
from sqlalchemy import create_engine


@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'Miguel'}  # fake user
    return render_template('index.html',
                           title='Home',
                           user=user)


@app.route('/db')
def cities_page():
    db = mdb.connect(user='danielj', host='localhost',
                     db='world',  charset='utf8')

    with db:
        cur = db.cursor()
        cur.execute('SELECT Name FROM City LIMIT 15;')
        query_results = cur.fetchall()
    cities = ''
    for result in query_results:
        cities += result[0]
        cities += '<br>'
    return cities


@app.route('/db_fancy')
def cities_page_fancy():
    db = mdb.connect(user='danielj', host='localhost',
                     db='world',  charset='utf8')

    with db:
        cur = db.cursor()
        cur.execute('SELECT Name, CountryCode, Population '
                    'FROM City ORDER BY Population LIMIT 15;')

        query_results = cur.fetchall()
    cities = []
    for result in query_results:
        cities.append(dict(name=result[0],
                      country=result[1], population=result[2]))
    return render_template('cities.html', cities=cities)


@app.route('/input')
def cities_input():
    return render_template('input.html')


def get_datapoints_from_sql(case, industry, geofips):
    engine = create_engine("mysql+mysqldb://danielj:@localhost/ecotest")
    con = engine.connect()

    query = 'SELECT x, y FROM fit_data WHERE '\
        'which_case="%s" and geofips="%s" and industry="%s" and method="%s"' %\
        (case, geofips, industry, 'AR_linreg')

    df = pd.read_sql(query, con=con.connection)

    return df.values


@app.route('/output')
def cities_output():
    industry = request.args.get('Industry')

    result_list = [{'name': 'City 1', 'recent': 100, 'forecast': 110},
                   {'name': 'City 2', 'recent': 100, 'forecast': 90}]

    data_input = get_datapoints_from_sql('full_data', 'manf', '25180')

    chart = pygal.XY(disable_xml_declaration=True, width=800, height=350)
    chart.title = 'Browser usage evolution'
    chart.x_labels = map(int, range(1965, 2025, 5))
    chart.add('Data', data_input)

    return render_template('output.html', industry=industry,
                           result_list=result_list, chart=chart)
