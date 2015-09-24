#!/usr/bin/python
import pandas as pd
import sys


def reformat_data(infile, outfile):
    df = pd.read_csv(infile)

    df_reduced = df[['line_id',
                     'line_desc',
                     'DataValue',
                     'GeoFips',
                     'GeoName',
                     'TimePeriod',
                     'CL_UNIT',
                     'UNIT_MULT']]

    df_reindex = df_reduced.set_index(['line_id',
                                       'line_desc',
                                       'GeoFips',
                                       'GeoName',
                                       'CL_UNIT',
                                       'UNIT_MULT',
                                       'TimePeriod'])

    df_unstack = df_reindex.unstack()

    df_unstack.columns = df_unstack.columns.get_level_values(1).values

    df_unstack = df_unstack.reset_index()

    df_unstack.to_csv(outfile)


def main(argv):
    infile1 = '../../data/bea-api_pulls/data/reginc_CA5.csv'
    outfile1 = '../../data/cleaned_data/regional_income_details_69-00.csv'
    reformat_data(infile1, outfile1)

    infile2 = '../../data/bea-api_pulls/data/reginc_CA5N.csv'
    outfile2 = '../../data/cleaned_data/regional_income_details_01-13.csv'
    reformat_data(infile2, outfile2)


if __name__ == "__main__":
        main(sys.argv[1:])
