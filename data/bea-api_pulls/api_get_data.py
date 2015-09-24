#!/usr/bin/python
import bea_api_wrapper as bea
#  import numpy as np
#  import pandas as pd
import sys


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


def main(argv):
    # get_regdata_pcpi()
    print '\n\n'
    # get_regprod_rpcgdp()
    get_reginc_data('CA5N')


if __name__ == "__main__":
    main(sys.argv[1:])
