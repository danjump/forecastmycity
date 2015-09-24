#!/usr/bin/python
import bea_api_wrapper as bea
#  import numpy as np
#  import pandas as pd
import sys


def get_parameter_lists(dataset):
    f = open('meta_data/params_list.txt', 'w+')

    # args dict valid args:
    # method, dataset, table, line, year, geo
    args_dict = {'method': 'GetParameterList', 'dataset': dataset}

    df = bea.api_pull_to_df(args_dict)

    print >>f, '\"%s\" Dataset Parameter List Summary:' % dataset
    for name, des in zip(df['ParameterName'], df['ParameterDescription']):
        print >>f, '%12s' % name, des
    print >>f, ''

    print >>f, '\"%s\" Dataset Parameter List full:' % dataset
    print >>f, df
    print >>f, ''
    print >>f, ''

    f.close()


def get_regprod_table_info():
    f = open('meta_data/table_info_RegionalProduct.txt', 'w+')

    args_dict = {'method': 'GetParameterValuesFiltered',
                 'dataset': 'RegionalProduct',
                 'targparam': 'Component'}

    df = bea.api_pull_to_df(args_dict)

    count1 = -1
    count1max = len(df)
    msa_good_count = 0
    msa_bad_count = 0
    for key, des in zip(df['Key'], df['Desc']):
        args_dict = {'method': 'GetParameterValuesFiltered',
                     'dataset': 'RegionalProduct',
                     'component': key,
                     'targparam': 'IndustryId'}

        count1 += 1

        df2 = bea.api_pull_to_df(args_dict)
        if df2 is not None:
            count2 = 0
            count2max = len(df2)
            local_bad_count = 0
            local_good_count = 0

            for key2, des2 in zip(df2['Key'], df2['Desc']):
                sys.stdout.write(
                    '\rTotalProgress: %d/%d  SubProgress %d/%d' % (
                        count1, count1max, count2, count2max) +
                    '   msa compatible:%d  noncompatible:%d  ' % (
                        msa_good_count, msa_bad_count))
                sys.stdout.flush()
                count2 += 1

                args_dict = {'method': 'GetData',
                             'dataset': 'RegionalProduct',
                             'component': key,
                             'industry': key2,
                             'geo': 'MSA',
                             'year': 'ALL'}
                df3 = bea.api_pull_to_df(args_dict)
                if df3 is not None:
                    msa_good_count += 1
                    local_good_count += 1
                    print >>f, 'For component \"%s\": %s' % (key, des)
                    print >>f, '  industry %s: %s' % (key2, des2)
                    print >>f, '  cols:', df3.columns
                    try:
                        df3_year = df3['TimePeriod'].unique().values
                    except:
                        df3_year = df3['TimePeriod'].unique()
                    print >>f, '  year entries: %d from %s to %s' % (
                        len(df3_year), df3_year[0], df3_year[len(df3_year)-1])
                    print >>f, ''
                else:
                    if local_bad_count > 5 and local_good_count < 6:
                        break
                    msa_bad_count += 1
                    local_bad_count += 1
    f.close()


def get_reginc_table_info():
    f = open('meta_data/RegionalIncome/full_table_info.txt', 'w+')
    f_temp = open('meta_data/RegionalIncome/line_lists.txt', 'w+')

    args_dict = {'method': 'GetParameterValuesFiltered',
                 'dataset': 'RegionalIncome',
                 'targparam': 'TableName'}

    df = bea.api_pull_to_df(args_dict)

    print df

    count1 = -1
    count1max = len(df)
    msa_good_count = 0
    msa_bad_count = 0
    for key, des in zip(df['Key'], df['Desc']):
        args_dict = {'method': 'GetParameterValuesFiltered',
                     'dataset': 'RegionalIncome',
                     'table': key,
                     'targparam': 'LineCode'}
        # args_dict['verbose'] = 1

        count1 += 1

        df2 = bea.api_pull_to_df(args_dict)

        print df2

        if df2 is not None:
            count2 = 0
            count2max = len(df2)
            local_bad_count = 0
            local_good_count = 0

            for key2, des2 in zip(df2['Key'], df2['Desc']):
                sys.stdout.write(
                    '\rTotalProgress: %d/%d  SubProgress %d/%d' % (
                        count1, count1max, count2, count2max) +
                    '   msa compatible:%d  noncompatible:%d  ' % (
                        msa_good_count, msa_bad_count))
                sys.stdout.flush()
                count2 += 1

                args_dict = {'method': 'GetData',
                             'dataset': 'RegionalIncome',
                             'table': key,
                             'line': key2,
                             'geo': 'MSA',
                             'year': 'ALL'}
                df3 = bea.api_pull_to_df(args_dict)
                if df3 is not None:
                    msa_good_count += 1
                    local_good_count += 1

                    print >>f_temp, '%12s %s' % (key2, des2)
                    print >>f, 'For table \"%s\": %s' % (key, des)
                    print >>f, '  line %s: %s' % (key2, des2)
                    print >>f, '  cols:', df3.columns
                    try:
                        df3_year = df3['TimePeriod'].unique().values
                    except:
                        df3_year = df3['TimePeriod'].unique()
                    print >>f, '  year entries: %d from %s to %s' % (
                        len(df3_year), df3_year[0], df3_year[len(df3_year)-1])
                    print >>f, ''
                else:
                    # if local_bad_count > 20 and local_good_count < 10:
                        # break
                    msa_bad_count += 1
                    local_bad_count += 1
    f.close()
    f_temp.close()


def get_regdata_table_info():
    f = open('meta_data/table_info_RegionalData.txt', 'w+')

    args_dict = {'method': 'GetParameterValuesFiltered',
                 'dataset': 'RegionalData',
                 'targparam': 'KeyCode'}

    df = bea.api_pull_to_df(args_dict)

    for key, des in zip(df['KeyCode'], df['Description']):
        args_dict = {'method': 'GetParameterValuesFiltered',
                     'dataset': 'RegionalData',
                     'keycode': key,
                     'targparam': 'Year'}
        df2 = bea.api_pull_to_df(args_dict)
        if df2 is not None:
                print >>f, 'For keycode \"%s\": %s' % (key, des)
                print >>f, '  cols:', df2.columns
                try:
                    df2_year = df2['TimePeriod'].unique().values
                except:
                    df2_year = df2['TimePeriod'].unique()
                print >>f, '  year entries: %d from %s to %s' % (
                    len(df2_year), df2_year[0], df2_year[len(df2_year)-1])
                print >>f, ''

    f.close()


def main(argv):
    '''
    regional_dataset_names = ['RegionalIncome',
                              'RegionalProduct',
                              'RegionalData']
    for dataset in regional_dataset_names:
        get_parameter_lists(dataset)
        '''
    if argv[0] == 'regprod':
        get_regprod_table_info()
    if argv[0] == 'reginc':
        get_reginc_table_info()
    if argv[0] == 'regdata':
        get_regdata_table_info()
    if argv[0] == 'all':
        get_regprod_table_info()
        get_reginc_table_info()
        get_regdata_table_info()


if __name__ == "__main__":
    main(sys.argv[1:])
