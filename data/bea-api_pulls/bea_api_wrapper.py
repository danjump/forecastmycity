import pandas as pd
import requests as rq
import json


def make_request(args_dict):
    base_url = 'http://bea.gov/api/data/?UserID='
    api_key = '2A4FC9C7-5523-4C19-B363-3AA5B5426339'
    url = base_url + api_key

    if 'method' in args_dict:
        url = url + '&method=' + args_dict['method']
    if 'dataset' in args_dict:
        url = url + '&datasetname=' + args_dict['dataset']
    if 'targparam' in args_dict:
        url = url + '&TargetParameter=' + args_dict['targparam']
    if 'parameter' in args_dict:
        url = url + '&ParameterName=' + args_dict['parameter']
    if 'table' in args_dict:
        url = url + '&TableName=' + args_dict['table']
    if 'component' in args_dict:
        url = url + '&Component=' + args_dict['component']
    if 'industry' in args_dict:
        url = url + '&IndustryId=' + args_dict['industry']
    if 'keycode' in args_dict:
        url = url + '&KeyCode=' + args_dict['keycode']
    if 'line' in args_dict:
        url = url + '&LineCode=' + args_dict['line']
    if 'year' in args_dict:
        url = url + '&Year=' + args_dict['year']
    if 'geo' in args_dict:
        url = url + '&GeoFips=' + args_dict['geo']

    url = url + '&ResultFormat=json'

    raw_json = json.loads(rq.get(url).content.decode('latin1'))

    return raw_json


def process_results(raw_json, args_dict):

    # json_request = raw_json['BEAAPI']['Request']
    json_result = raw_json['BEAAPI']['Results']

    if 'verbose' in args_dict:
        if args_dict['verbose'] >= 2:
            print '\nPull Results:\n%s\n\n' % json_result

    if args_dict['method'] == 'GetParameterList':
        key = 'Parameter'
    if args_dict['method'] == 'GetParameterValues':
        key = 'ParamValue'
    if args_dict['method'] == 'GetParameterValuesFiltered':
        key = 'ParamValue'
    if args_dict['method'] == 'GetData':
        key = 'Data'

    try:
        df = pd.DataFrame(json_result[key])
    except:
        if 'verbose' in args_dict:
            if args_dict['verbose'] >= 1:
                print 'ERROR READING PULL!-------------------------------------'
                print 'Pull Results:\n%s\n' % json_result

        df = None

    return df


def api_pull_to_df(args_dict):
    raw_json = make_request(args_dict)

    df = process_results(raw_json, args_dict)

    return df
