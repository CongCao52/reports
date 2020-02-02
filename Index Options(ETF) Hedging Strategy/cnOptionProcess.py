# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:23:14 2019

@author: DELL
"""

import pandas as pd
import numpy as np

call_ticker=pd.read_excel('callTicker.xlsx')
put_ticker=pd.read_excel('putTicker.xlsx')
option_all=pd.read_excel('OptionPriceCN.xlsx')

call_ticker.columns=['时间','代码']
put_ticker.columns=['时间','代码']
option_all['时间']=option_all['时间'].apply(str.replace,args=('/','-'))


call=pd.merge(call_ticker,option_all,on=['时间','代码'],how='left')
put=pd.merge(put_ticker,option_all,on=['时间','代码'],how='left')

call.to_excel('CNcallPrice.xlsx')
put.to_excel('CNputPrice.xlsx')

call_final=call[['开盘价(元)','收盘价(元)','最高价(元)','最低价(元)','成交量']]
put_final=put[['开盘价(元)','收盘价(元)','最高价(元)','最低价(元)','成交量']]

call_final=call_final.fillna(method='ffill')
put_final=put_final.fillna(method='ffill')

call_final.columns=['open','close','high','low','volume']
put_final.columns=['open','close','high','low','volume']

call_final.index=call['时间']
put_final.index=put['时间']

call_final.to_excel('data_callCN.xlsx')
put_final.to_excel('data_putCN.xlsx')

data_optCN=pd.concat((call_final,put_final),axis=1)