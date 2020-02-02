# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:27:12 2019

@author: DELL
"""

import pandas as pd
import numpy as np

read_path='.\spyOption\\'
save_path='.\data\\'

dateList=['201606','201607','201608','201609','201610','201611','201612',\
          '201701','201702','201703','201704','201705','201706','201707',\
          '201708','201709','201710','201711','201712','201801','201802',\
          '201803','201804','201805','201806','201807','201809','201810',\
          '201811','201812','201901','201902','201903','201904','201905']

data=pd.DataFrame()

for date in dateList:
    temp_data=pd.read_csv(read_path+'CBOE_O_SPY_'+date+'_1day.csv')
    data=pd.concat((data,temp_data))
    
data.to_csv('OptionPriceUS.csv')

dates=np.sort(list(set(data['date'])))

call_data=data[data['OptionType']=='call'].reset_index()
put_data=data[data['OptionType']=='put'].reset_index()

call=pd.DataFrame()
put=pd.DataFrame()

for date in dates:
    
    print(date)
    
    temp_call=call_data[call_data['date']==date]
    if max(temp_call['volume'])==0:
        call_ticker=pd.DataFrame(temp_call.loc[temp_call.index[0]+int(temp_call.shape[0]/2),:]).T
    else:
        call_ticker=temp_call[temp_call['volume']==max(temp_call['volume'])]
    temp_put=put_data[put_data['date']==date]
    if max(temp_put['volume'])==0:
        put_ticker=pd.DataFrame(temp_put.loc[temp_put.index[0]+int(temp_put.shape[0]/2),:]).T
    else:
        put_ticker=temp_put[temp_put['volume']==max(temp_put['volume'])]
    
    call=pd.concat((call,call_ticker),axis=0)
    put=pd.concat((put,put_ticker),axis=0)

call_final=call[['date','close2','volume']]
put_final=put[['date','close2','volume']]

call_final['open']=call_final['low']=call_final['high']=call_final['close2']
put_final['open']=put_final['low']=put_final['high']=put_final['close2']
call_final.rename(columns={'close2':'close'},inplace=True)
put_final.rename(columns={'close2':'close'},inplace=True)
call_final.index=call_final['date']
put_final.index=put_final['date']

data_optionUS=pd.concat((call_final[['open','high','low','close','volume']],\
                         put_final[['open','high','low','close','volume']]),axis=1)
data_optionUS.index=pd.to_datetime(data_optionUS.index,format='%Y%m%d')
data_optionUS.to_excel(save_path+'data_optUS.xlsx')