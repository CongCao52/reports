import os
import pandas as pd
import numpy as np
from datetime import datetime


data_path = 'F:\BU work\MF803\project\data'
os.chdir(data_path)
file_list = os.listdir()
first_file = 'L2_options_20170403.csv'


contract_list = ['XLB170616C00054000',
'XLB170519C00050000', 'XLB170519C00051000', 'XLB170519C00052000',
'XLB170519C00053000', 'XLB170519C00054000', 'XLB170616C00050000',
'XLB170616C00051000', 'XLB170616C00052000', 'XLB170616C00053000']

reader1 = pd.read_csv('%s\%s' % (data_path, first_file), index_col='OptionSymbol',
                      usecols=['UnderlyingSymbol', 'UnderlyingPrice', 'OptionSymbol', 'Type', 'Expiration',
                               'DataDate', 'Strike', 'Bid', 'Ask'])


def extract_train(file_list=file_list):
    start = datetime.strftime(datetime.strptime('2017-4-15', '%Y-%m-%d'), '%m/%d/%Y')
    end = datetime.strftime(datetime.strptime('2017-06-01', '%Y-%m-%d'), '%m/%d/%Y')
    for file_name in file_list:
        reader = pd.read_csv('%s\%s' % (data_path, file_name), index_col='OptionSymbol',
                             usecols=['UnderlyingSymbol', 'UnderlyingPrice', 'OptionSymbol', 'Type',
                                      'Expiration', 'DataDate', 'Strike', 'Bid', 'Ask'])
        train_file = reader.loc[reader['UnderlyingSymbol'] == 'XLB']
        spot = train_file.head(1).loc[:, 'UnderlyingPrice'].tolist()[0]
        upper = 1.1 * spot
        lower = 0.9 * spot
        train_data = train_file.loc[train_file['Strike'] <= upper]
        train_data = train_data.loc[train_data['Strike'] >= lower]
        train_data['Bid'] = train_data['Bid'].replace(0, np.nan)
        train_data = train_data.dropna(how='any')
        mid = (train_data['Bid'] + train_data['Ask'])/2
        train_data = pd.concat([train_data, mid], axis=1)
        col_name = train_data.columns.tolist()
        col_name[-1] = 'Mid'
        train_data.columns = col_name
        train_data = train_data.loc[train_data['Expiration'] <= end]
        train_data = train_data.loc[train_data['Expiration'] >= start]
        train_data.to_csv('%s\XLB_%s_train.csv' % (data_path, file_name[11:19]))
        print('read %s' % file_name)


def extract_contract(contract_list=contract_list, file_list=file_list):
    for contract in contract_list:
        price = []
        time = []
        expire = datetime.strftime(datetime.strptime(reader1.loc[contract, 'Expiration'], '%m/%d/%Y'), '%Y%m%d')
        strike = reader1.loc[contract, 'Strike']
        for file_name in file_list:
            if expire >= file_name[11:19]:
                reader = pd.read_csv('%s\%s' % (data_path, file_name), index_col='OptionSymbol',
                                     usecols=['UnderlyingSymbol', 'OptionSymbol', 'Type',
                                              'Expiration', 'DataDate', 'Strike', 'Bid', 'Ask'])
                cur_time = datetime.strftime(datetime.strptime(reader.loc[contract, 'DataDate'], '%m/%d/%Y'), '%Y-%m-%d')
                price.append((reader.loc[contract, 'Bid']+reader.loc[contract, 'Ask'])/2)
                time.append(cur_time)
            print('read %s' % file_name)
        expire_col = [datetime.strftime(datetime.strptime(expire, '%Y%m%d'), '%Y-%m-%d')]*len(price)
        strike_col = [strike]*len(price)
        df = pd.DataFrame([expire_col, price, strike_col], columns=time, index=['Expiration', 'Mid', 'Strike'])
        df = df.T
        df.to_csv('%s.csv' % contract)


# extract_train()
# extract_contract()
