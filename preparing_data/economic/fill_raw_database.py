#!/usr/bin/python

import sys
import getopt
import pandas as pd
# import MySQLdb as mdb
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "")
    except getopt.GetoptError:
        print 'Error: bad input arguments'
        sys.exit(2)

    for arg, i in zip(args, range(len(args))):
        print 'Arg %d is ' % (i), arg

    infiles = dict()
    infiles['rpi'] = "/home/danielj/insight/project/data/"\
        "BEA-RealPersonalIncome/RPI_MSA_2008_2013_nofooter.csv"
    infiles['rgdp'] = "/home/danielj/insight/project/data/"\
        "BEA-GDP_MetroArea/gmpRGDP.csv"
    infiles['labor_data_90_94'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.data.0.CurrentU90-94.csv"
    infiles['labor_data_95_99'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.data.0.CurrentU95-99.csv"
    infiles['labor_data_00_04'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.data.0.CurrentU00-04.csv"
    infiles['labor_data_05_09'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.data.0.CurrentU05-09.csv"
    infiles['labor_data_10_14'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.data.0.CurrentU10-14.csv"
    infiles['labor_data_15_19'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.data.0.CurrentU15-19.csv"
    infiles['labor_legend_measure'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.measure.csv"
    infiles['labor_legend_period'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.period.csv"
    infiles['labor_legend_series'] = "/home/danielj/insight/project/data/"\
        "BLS-EmploymentLabor/csvs/la.series.csv"

    print infiles
    for key in infiles:
        read_file_write_db(infiles[key], key)
# end main()


def read_file_write_db(infile, key):
    try:
        print 'Reading file %s' % infile
        df = pd.read_csv(infile)
        print df.head()
    except:
        print 'Error reading file:', sys.exc_info()[0]
        return 0
    try:
        print 'Connecting to database...'
        # con = mdb.connect('localhost', 'danielj', '', 'ecotest')
        engine = create_engine(
            "mysql+mysqldb://danielj:@localhost/ecotest")
    except:
        print 'Error connecting to db:', sys.exc_info()[0]
        return 0
    try:
        print 'Writing to database...'
        df.to_sql(con=engine, name=key,
                  if_exists='replace', flavor='mysql')
    except OperationalError as err:
        print 'Error writing to db:', sys.exc_info()[0]
        print err.message
        return 0
# end read_file()

if __name__ == "__main__":
    main(sys.argv[1:])
