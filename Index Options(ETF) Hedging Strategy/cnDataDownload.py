# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:53:39 2019

@author: DELL
"""
import numpy as np
import pandas as pd
from WindPy import *
w.start()


'''data Download'''

save_path='.\data\\'

ticker_CN=['000016.SH','950077.CSI','601318.SH','600519.SH','600036.SH',\
           '600887.SH','600276.SH','600048.SH','600030.SH','600585.SH',\
           '600031.SH','600104.SH']

name_CN=['上证50','上证50反向','中国平安','贵州茅台','招商银行','伊利股份',\
         '恒瑞医药','保利地产','中信证券','海螺水泥','三一重工','上汽集团']
 
start=["2001-01-01","2007-01-01"]
end=["2006-12-31","2019-05-31"]
feature=['open','high','low','close','volume']

stock_CN=pd.DataFrame()
data_CN=pd.DataFrame()

for i in range(len(name_CN)):
    if i >=2:
        stockCN_temp=w.wsd(ticker_CN[i], "open,high,low,close,volume",\
                           start[0], end[0], "PriceAdj=F")
        stockCN_temp = pd.DataFrame(stockCN_temp.Data,index=feature,\
                                    columns=stockCN_temp.Times).T
        stock_CN=pd.concat((stock_CN,stockCN_temp),axis=1)
        
    
    
    dataCN_temp=w.wsd(ticker_CN[i],"open,high,low,close,volume",\
                      start[1], end[1], "PriceAdj=F")
    dataCN_temp = pd.DataFrame(dataCN_temp.Data,index=feature,\
                               columns=dataCN_temp.Times).T 
    data_CN=pd.concat((data_CN,dataCN_temp),axis=1)




stock_CN.to_excel(save_path+'stock_CN.xlsx')
data_CN.to_excel(save_path+'data_CN.xlsx')