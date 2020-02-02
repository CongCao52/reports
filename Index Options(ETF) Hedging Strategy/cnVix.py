# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:22:43 2019

@author: DELL
"""

import numpy as np
import pandas as pd
from WindPy import *
w.start()

def vix_calculation(T,strike_list,option_call,option_put):
    
    
    strikeFloat_list=[float(x) for x in strike_list]
    
    option_call.columns=strike_list
    option_put.columns=strike_list

    
    option_diff=abs(option_call-option_put)
    minDiff_idx=option_diff.apply(np.argsort,1).iloc[:,0]
        
    S=strike_list[minDiff_idx[0]]
    S_float=float(S)
    

    callPut_minDiff=np.array(option_call[S]-option_put[S])
    callPut_minDiff=pd.DataFrame(callPut_minDiff)
    callPut_minDiff=callPut_minDiff.iloc[0,0]


    
    F=np.array(S_float)+np.exp(r*T)*abs(callPut_minDiff)
    
    Fstrike_diff=strikeFloat_list-F   
    Fstrike_diff=[x for x in Fstrike_diff if x<0]
    

    K0_index=np.argsort(Fstrike_diff)[-1]
    K0=strikeFloat_list[K0_index]
    
    price_index=pd.DataFrame(np.array(strikeFloat_list)-K0,\
                             index=strike_list,columns=option_call1.index).T
    
    price_call=option_call[price_index>0].fillna(0)
    price_put=option_put[price_index<0].fillna(0)
    price_K0=((option_call[price_index==0]+option_put[price_index==0])/2).fillna(0)
    
    price=price_call+price_put+price_K0

    K=pd.DataFrame(strikeFloat_list,index=strike_list,columns=option_call1.index).T
    
    delta_k=[strikeFloat_list[1]-strikeFloat_list[0]]+[(strikeFloat_list[i+1]-strikeFloat_list[i-1])/2 \
                for i in range(1,len(strikeFloat_list)-1)]+[strikeFloat_list[-1]-strikeFloat_list[-2]]
    delta_k=pd.DataFrame(delta_k,index=strike_list,columns=option_call1.index).T
    

    var=2/T*np.exp(r*T)*(price*delta_k/K**2).sum(1)-1/T*(F/K0-1)**2
    
    return var

if __name__=="__main__":

    save_path='.\data\\'
    
    days=365*24*60
    dayM=30*60*24
    r=0
    
    index_name=['ticker','type','strike','month','expireDate']

    dates=pd.DataFrame()
    vix_us=pd.read_excel(save_path+'data_VIX.xlsx')['close']
    dates['Date']=vix_us.index
    dates=dates['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    dates=dates[dates>='2019-05-30']#2015-02-09
    ivix=pd.DataFrame()

    for i in range(len(dates)):
        
        date=dates.iloc[i]
        print(date)
        
        option_data=w.wset("optionchain","date="+date+";us_code=510050.SH;option_var=全部;\
               call_put=全部;field=option_code,call_put,strike_price,month,expiredate")
        
        data=pd.DataFrame(option_data.Data,index=index_name).T
    
        month_list=list(set(data['month']))
        month_list.sort()
        
        call1=data[(data['month']==month_list[0]) & (data['type']=='认购')]
        put1=data[(data['month']==month_list[0]) & (data['type']=='认沽')]
        call2=data[(data['month']==month_list[1]) & (data['type']=='认购')]
        put2=data[(data['month']==month_list[1]) & (data['type']=='认沽')]
        
        data_call1=w.wsd(list(call1['ticker']), "close", date, date, "PriceAdj=F")
        data_put1=w.wsd(list(put1['ticker']), "close", date, date, "PriceAdj=F")
        data_call2=w.wsd(list(call2['ticker']), "close", date, date, "PriceAdj=F")
        data_put2=w.wsd(list(put2['ticker']), "close", date, date, "PriceAdj=F")
        
        
        option_call1=pd.DataFrame(data_call1.Data,index=data_call1.Times,\
                                 columns=list(call1['ticker']))
        option_put1=pd.DataFrame(data_put1.Data,index=data_put1.Times,\
                                 columns=list(put1['ticker']))
        
        option_call2=pd.DataFrame(data_call2.Data,index=data_call2.Times,\
                                 columns=list(call2['ticker']))
        option_put2=pd.DataFrame(data_put2.Data,index=data_put2.Times,\
                                 columns=list(put2['ticker']))
    
        T1=call1['expireDate'].iloc[0]*60*24/days
        T2=call2['expireDate'].iloc[0]*60*24/days
    
        strike_list1=list(call1['strike'])#list(call1['ticker'].apply(lambda x: x[9:13]))
        strike_list2=list(call2['strike'])#list(call2['ticker'].apply(lambda x: x[9:13]))

        if T1>0:        
            var1=vix_calculation(T1,strike_list1,option_call1,option_put1)
            
            var2=vix_calculation(T2,strike_list2,option_call2,option_put2)
        
            temp_ivix=100*np.sqrt((T1*var1*(T2*days-dayM)/(T2*days-T1*days)+\
                             T2*var2*(dayM-T1*days)/(T2*days-T1*days))*days/dayM)

        else:
            var2=vix_calculation(T2,strike_list2,option_call2,option_put2)            
            
            temp_ivix=100*np.sqrt(var2)
        
        ivix=pd.concat((ivix,temp_ivix))

    ivix.columns=['ivix']
#    ivix.to_excel(save_path+'data_IVIX.xlsx')





