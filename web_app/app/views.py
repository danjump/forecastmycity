import MySQLdb as mdb
import pandas as pd
from flask import render_template, request
from app import app
from a_Model import ModelIt


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


@app.route('/output')
def cities_output():
    db = mdb.connect(user='danielj', host='localhost',
                     db='ecotest',  charset='utf8')
    # pull 'ID' from input field and store it
    city = request.args.get('ID')

    with db:
        # just select the city from the world_innodb that the user inputs
        sql_query = 'SELECT gdp.2001, gdp.2002, gdp.2003, gdp.2004, gdp.2005, '\
                    'gdp.2006, gdp.2007, gdp.2008, gdp.2009, gdp.2010, '\
                    'gdp.2011, gdp.2012, gdp.2013, region '\
                    'FROM norm_gdp_per_msa_all_industry gdp '\
                    'WHERE region LIKE \'%%%s%%\';' % city
        df = pd.read_sql(sql_query, con=db)

    cities = df.to_dict(orient='records')

    # call a function from a_Model package.
    # note we are only pulling one result in the query
    pop_input = cities[0]['2008']
    the_result = ModelIt(city, pop_input)
    return render_template('output.html', cities=cities,
                           the_result=the_result)
