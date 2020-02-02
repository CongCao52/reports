# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:02:48 2019

@author: DELL
"""

from pandas_datareader import data as pdr
import pandas as pd

class DataProcess():
    
    def __init__(self):
        
        self.tickers=[]
        
    def getStockData(self,tickers, st, ed):
        
        """ Get data from yahoo Finance"""
        data = pdr.get_data_yahoo(tickers, start = st, end = ed)[['High','Low',\
                                 'Open','Adj Close','Volume']]
        data.columns=['high','low','open','close','volume']
        return data
    
    
if __name__=="__main__":

    save_path='.\data\\'
    
    tickers=['SPY','SH','MSFT','AAPL','BA','JPM','JNJ',\
           'XOM','T','PG','DIS','KO']
    tickers2=['MSFT','AAPL','BA','JPM','JNJ',\
           'XOM','T','PG','DIS','KO']
    
    start=["2001-01-01","2007-01-01"]
    end=["2006-12-31","2019-05-31"]
    option=['stock_US','data_US']
    
    data=pd.DataFrame()
    dataProcessor=DataProcess()
    for ticker in tickers2:
        temp_data = dataProcessor.getStockData(ticker,start[0],end[0])
        data=pd.concat((data,temp_data),axis=1)
    
#    vix = dataProcessor.getStockData('^VIX',start,end)    
        
    data.to_excel(save_path+option[0]+'.xlsx')
#    vix.to_excel(save_path+'data_VIX.xlsx')