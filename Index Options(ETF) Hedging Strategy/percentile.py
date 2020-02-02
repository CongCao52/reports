# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:27:44 2019

@author: DELL
"""

import pandas as pd
import numpy as np

save_path='.\data\\'

name=['data_CN','data_US']
name2=['CN','US']
opt=1
N=3
data=pd.read_excel(save_path+name[opt]+'.xlsx')

percentile=pd.DataFrame()


peak=data['close'][0]
trough=data['close'][0]

for i in range(252*N,data.shape[0]):

    peak=max(data['close'][i-252*N:i+1])
    trough=min(data['close'][i-252*N:i+1])
    

    percentile.loc[i,'percentage']=(data['close'][i]-trough)/(peak-trough)
        
percentile.index=data.index[252*N:]
percentile.dropna(inplace=True)
percentile.to_excel(save_path+'percentile_'+name2[1]+'.xlsx')