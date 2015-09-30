import MySQLdb as mdb
import pandas as pd
import numpy as np
import pygal
from flask import render_template, request
from app import app


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


def read_results_shelf(infile):
    import shelve
    readshelf = shelve.open(infile, 'r')
    indata = readshelf['args_dict']

    readshelf.close()

    result_dict = {'data_x': indata['train_years'],
                   'data_y': indata['train_values'],
                   'pred_y': indata['train_pred'],
                   'score': indata['train_score'],
                   'forecast_x': indata['proj_years'],
                   'forecast_y': indata['proj_pred']}
    return result_dict


@app.route('/output')
def cities_output():
    industry = request.args.get('Industry')

    result_list = [{'name': 'City 1', 'recent': 100, 'forecast': 110},
                   {'name': 'City 2', 'recent': 100, 'forecast': 90}]

    infile = '../analysis/results/regional_income/backup/'\
        'results_g12020_irettrd_w4_t0.shelf'

    results = read_results_shelf(infile)

    x_range = np.append(results['data_x'], results['forecast_x'])
    x_range = x_range.astype(int)
    data_df = pd.DataFrame({'year': results['data_x'],
                            'values': results['data_y']})

    data_input = []
    for yr, val in zip(data_df['year'].values, data_df['values'].values):
        data_input.append((yr, val))

    chart = pygal.XY(disable_xml_declaration=True, width=800, height=350)
    chart.title = 'Browser usage evolution'
    chart.x_labels = map(int, range(1965, x_range[-1]+5, 5))
    chart.add('Data', data_input)

    return render_template('output.html', industry=industry,
                           result_list=result_list, chart=chart)
