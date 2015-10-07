#!/usr/bin/python
import bea_api_wrapper as bea
#  import numpy as np
#  import pandas as pd
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError


def get_regdata_pcpi():

    # get personal income per capita data for msa's
    args_dict = {'method': 'GetData',
                 'dataset': 'RegionalData',
                 'keycode': 'PCPI_MI',
                 'year': 'ALL',
                 'geo': 'MSA'}

    df = bea.api_pull_to_df(args_dict)

    if df is not None:
        print df
        outfile = 'data/regdata_pcpimsa.csv'
        df.to_csv(outfile)


def get_reginc_data(table):

    args_dict = {'method': 'GetParameterValuesFiltered',
                 'dataset': 'RegionalIncome',
                 'targparam': 'LineCode',
                 'table': table}
    args_dict['verbose'] = 1

    line_df = bea.api_pull_to_df(args_dict)

    if line_df is not None:
        df = None

        for line, desc in zip(line_df['Key'], line_df['Desc']):
            # get personal income per capita data for msa's
            args_dict = {'method': 'GetData',
                         'dataset': 'RegionalIncome',
                         'table': table,
                         'year': 'ALL',
                         'geo': 'MSA',
                         'line': line}
            args_dict['verbose'] = 1

            tmp_df = bea.api_pull_to_df(args_dict)

            if tmp_df is not None:
                print tmp_df.head(1)
                tmp_df['line_id'] = line
                tmp_df['line_desc'] = desc
                if df is None:
                    df = tmp_df
                else:
                    df = df.append(tmp_df, ignore_index=True)
                print tmp_df.head(1)
                print ''
                outfile_tmp = 'data/reginc_seperate/reginc_t%s_l%s.csv' % (
                    table, line)
                tmp_df.to_csv(outfile_tmp)
            else:
                print 'BAD TMP DF! %s' % desc

        outfile = 'data/reginc_%s.csv' % table
        df.to_csv(outfile)

        return df


def get_regprod_rpcgdp():

    args_dict = {'method': 'GetParameterValuesFiltered',
                 'dataset': 'RegionalProduct',
                 'targparam': 'IndustryId',
                 'component': 'PCRGDP_MAN'}
    args_dict['verbose'] = 1

    ind_df = bea.api_pull_to_df(args_dict)

    if ind_df is not None:
        df = None

        for industry, desc in zip(ind_df['Key'], ind_df['Desc']):
            # get personal income per capita data for msa's
            args_dict = {'method': 'GetData',
                         'dataset': 'RegionalProduct',
                         'component': 'PCRGDP_MAN',
                         'year': 'ALL',
                         'geo': 'MSA',
                         'industry': industry}
            args_dict['verbose'] = 1

            tmp_df = bea.api_pull_to_df(args_dict)

            if tmp_df is not None:
                print tmp_df.head(1)
                tmp_df['ind_id'] = industry
                tmp_df['ind_desc'] = desc
                if df is None:
                    df = tmp_df
                else:
                    df = df.append(tmp_df, ignore_index=True)
                print tmp_df.head(1)
                print ''
            else:
                print 'BAD TMP DF! %s' % desc

        outfile = 'data/regdata_pcrgdp.csv'
        df.to_csv(outfile)


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
    # get_regdata_pcpi()
    print '\n\n'
    # get_regprod_rpcgdp()
    # get_reginc_data('CA5N')
    dfn = get_reginc_data('CA25N')
    df = get_reginc_data('CA25')

    print len(dfn), len(df)

    bothdf = pd.concat([df, dfn], axis=0, ignore_index=True)

    print len(bothdf), bothdf.columns

    write_to_sql(bothdf, 'employment_dataset', 'forecastmycity', 'replace')


if __name__ == "__main__":
    main(sys.argv[1:])
