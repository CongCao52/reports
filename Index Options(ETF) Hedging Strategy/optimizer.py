# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:48:46 2019

@author: DELL
"""

import numpy as np
import scipy.optimize as opt

class optimizer(object):
    
    def __init__(self):
        pass
    
    def optimize(self,rtn,risk_averse,cov):
        
        w0=np.random.randn(1,len(rtn))#initial weight
        #constraints for optimization
        constrain = ({'type': 'eq', 'fun': lambda x:  -sum(x)+1},\
                     {'type': 'ineq', 'fun': lambda x: x},\
                     {'type': 'ineq', 'fun': lambda x: -x+0.1})
        coeffs = [rtn,risk_averse,cov]
        options={'disp':False}
        result = opt.minimize(self.objective,w0,args=coeffs,options = options,\
                               constraints=constrain,method="SLSQP")
        w_opt=result.x
        return w_opt
        
    def objective(self,w,coeffs):
        '''objective function'''
        term1=np.matmul(w,coeffs[0])
        term2=coeffs[1]*np.matmul(np.matmul(w,coeffs[2]),np.transpose(w))/2
        
        return term2-term1