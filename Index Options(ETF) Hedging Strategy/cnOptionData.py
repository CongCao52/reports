# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:32:35 2019

@author: DELL
"""

import numpy as np
import pandas as pd
from WindPy import *
w.start()

if __name__=="__main__":

    save_path='.\data\\'
    
    index_name=['ticker','type','strike','month']    

    dates=pd.DataFrame()
    vix_us=pd.read_excel(save_path+'data_VIX.xlsx')['close']
    dates['Date']=vix_us.index
    dates=dates['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    dates=dates[dates>='2019-04-18']#2015-02-09~2019-05-31
    
    call_final=pd.DataFrame()
    put_final=pd.DataFrame()
    

    for i in range(len(dates)):
        
        date=dates.iloc[i]
        print(date)
        
        option_data=w.wset("optionchain","date="+date+";us_code=510050.SH;option_var=全部;\
               call_put=全部;field=option_code,call_put,strike_price,month")
        
        data=pd.DataFrame(option_data.Data,index=index_name).T
    
        month_list=list(set(data['month']))
        month_list.sort()
        
        call=data[(data['month']==month_list[0]) & (data['type']=='认购')]
        put=data[(data['month']==month_list[0]) & (data['type']=='认沽')]
        
        idx_call=call.index[0]+call.shape[0]//2
        idx_put=put.index[0]+put.shape[0]//2
        call_final.loc[i,'code']=call.loc[idx_call,'ticker']
        put_final.loc[i,'code']=put.loc[idx_put,'ticker']
        
    call_final.index=dates
    put_final.index=dates
    
#    call_final.to_excel('callTicker.xlsx')
#    put_final.to_excel('putTicker.xlsx')    
        
        