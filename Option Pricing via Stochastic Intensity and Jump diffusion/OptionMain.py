# -*- coding: utf-8 -*-

import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt,log,exp
from scipy.stats import norm
from scipy.optimize import brute, fmin, fsolve
from BCC_option_valuation import BCC_value, H93_value
from datetime import datetime 

i1 = 0
i2 = 0
i3 = 0
min_MSE1 = 500
min_MSE2 = 5000
min_MSE3 = 5000

class Option():
    def __init__(self, code, S0, K, t, r, M, v0, close, opt_type):
        '''
        Input：
        code: str
            Contract code
        S0: float
            Spot price
        K: float
            Strike price
        t: Datetime
            Datatime
        r: float
            Risk-free rate
        M: Datetime
            Maturity
        v0: float
            Initial variance
        close: float
            Market price of option
        opt_type: string
            Call/Put
        '''
        self.code = code
        self.S0 = S0 
        self.K = K 
        self.t = t 
        self.r = r 
        self.M = M 
        self.v0 = v0 
        self.close = close 
        self.type = opt_type 
        self.update_ttm()
        
    def __str__(self):
        tp = 'Call option' if self.type == 'call' else 'Put Option'
        str1 = 'Contract code: {0} \n Spot price: {1} \n Strike: {2} \n'.format(self.code, self.S0, self.K) 
        str2 = 'Datadate: {0} \n risk-free rate: {1} \n Maturity: {2} \n Option price: {3} \n Option type: {4}'.format(self.t, self.r, self.M, self.close, tp)
        str3 = '\n Initial variance: {0} \n BS model price: {1}'.format(self.v0, self.opt_bs_value())
        basic_info = str1 + str2 + str3
        return basic_info
        
    def update_ttm(self): 
        ''' Updates time-to-maturity self.T. ''' 
        if self.t > self.M: 
            raise ValueError("Pricing date later than maturity.") 
        self.T = (self.M - self.t).days / 365.
        
    def d1(self): 
        ''' Helper function. ''' 
        self.update_ttm()
        d1 = ((log(self.S0 / self.K)
            + (self.r + 0.5 * self.sigma() ** 2) * self.T) 
            / (self.sigma() * sqrt(self.T))) 
        return d1
    
    def sigma(self):
        return sqrt(self.v0)
    
    def opt_bs_value(self):
        '''BS model price'''
        self.update_ttm()
        d1 = self.d1() 
        d2 = ((log(self.S0 / self.K) 
            + (self.r - 0.5 * self.sigma() ** 2) * self.T) 
            / (self.sigma() * sqrt(self.T))) 
        call_value = (self.S0 * norm.cdf(d1, 0.0, 1.0) 
            - self.K * exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0)) 
        if self.type == 'call':
            return call_value
        else:
            put_value = max(0, call_value + self.K*exp(-self.r*self.T) - self.S0)
            return put_value
    def opt_bs_delta(self):
        '''Calculating delta of BS model'''
        self.update_ttm()
        d1 = self.d1() 
        if self.type == 'call':
            return norm.cdf(d1, 0.0, 1.0)
        else:
            return -norm.cdf(-d1, 0.0, 1.0)
        
    def opt_H93_value(self, kappa_v, theta_v, sigma_v, rho, v0):
        '''
        H93 model price
        (S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, tp)
        '''        
        self.update_ttm()
        return H93_value(self.S0, self.K, self.T, self.r, 
                         kappa_v, theta_v, sigma_v, rho, v0, self.type)
        
    def opt_BCC_value(self, kappa_v, theta_v, sigma_v, rho, v0,
                      lamb, mu, delta):
        '''
        BCC model price
        (S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, tp)
        '''
        self.update_ttm()
        return BCC_value(self.S0, self.K, self.T, self.r, 
                         kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta, self.type)
        
    def imp_vol(self, sigma_est = 0.2):
        '''Implied volatility by BS model'''
        opt = Option(self.code, self.S0, self.K, self.t, self.r, self.M, self.v0,
                     self.close, self.type)
        opt.update_ttm()
        def diff(sigma):
            opt.v0 = sigma * sigma
            return opt.opt_bs_value() - opt.close
        iv = max(0,fsolve(diff, sigma_est)[0])
        self.iv = iv
        return iv
        
#==============================================================================
'''
Calibration SV
'''

def H93_error_function(p0): 
    ''' Error function for parameter calibration in BCC97 model via 
    Lewis (2001) Fourier approach.
    
    Parameters 
    ========== 
    kappa_v: float 
        mean-reversion factor 
    theta_v: float 
        long-run mean of variance 
    sigma_v: float 
        volatility of variance 
    rho: float 
        correlation between variance and stock/index level 
    v0: float 
        initial, instantaneous variance
    
    Returns
    ======= 
    MSE: float
        mean squared error 
    ''' 
    global i1, min_MSE1 
    kappa_v, theta_v, sigma_v, rho, v0 = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or rho < -1.0 or rho > 1.0: 
        return 500.0 
    if 2 * kappa_v * theta_v < sigma_v ** 2: 
        return 500.0 
    se = [] 
    for option in options: 
        model_value = option.opt_H93_value(kappa_v, theta_v, sigma_v, rho, v0) 
        se.append((model_value - option.close) ** 2) 
    MSE = sum(se) / len(se) 
    min_MSE1 = min(min_MSE1, MSE) 
    if i1 % 25 == 0: 
        print('%4d |' % i1, np.array(p0), '| %11.7f | %11.7f' % (MSE, min_MSE1)) 
    i1 += 1 
    return MSE


def H93_calibration_full():
    ''' Calibrates H93 stochastic volatility model to market quotes. ''' 
    # first run with brute force 
    # (scan sensible regions)
    p0 = brute(H93_error_function, 
               ((2.5, 10.6, 5.0), # kappa_v 
                (0.01, 0.041, 0.01), # theta_v 
                (0.05, 0.251, 0.1), # sigma_v 
                (-0.75, 0.01, 0.25), # rho 
                (0.01, 0.031, 0.01)), # v0 
                finish=None)
               
    # second run with local, convex minimization
    # (dig deeper where promising) 
    opt = fmin(H93_error_function, p0, 
               xtol=0.000001, ftol=0.000001,
               maxiter=750, maxfun=900)
    return opt
#==============================================================================
    
'''
Calibration of jump-diffusion
'''

def BCC_error_function(p0): 
    ''' Error function for parameter calibration in M76 Model via Carr-Madan (1999) FFT approach.
    
    Parameters 
    ========== 
    lamb: float 
        jump intensity
    mu: float 
        expected jump size 
    delta: float 
        standard deviation of jump

    Returns 
    =======
    MSE: float 
        mean squared error 
    ''' 
    # 
    # Initial Parameter Guesses 
    # 
    
    global i2, min_MSE2
    lamb, mu, delta = p0 
    if lamb < 0.0 or mu < -0.6 or mu > 0.0 or delta < 0.0: 
        return 5000.0 
    se = [] 
    for option in options: 
        model_value = option.opt_BCC_value(kappa_v, theta_v, sigma_v, rho, v0,
                      lamb, mu, delta) 
        se.append((model_value - option.close) ** 2) 
    MSE = sum(se) / len(se) 
    min_MSE2 = min(min_MSE2, MSE) 
    if i2 % 25 == 0: 
        print('%4d |' % i2, np.array(p0), '| %11.7f | %11.7f' % (MSE, min_MSE2))
    i2 += 1 
    if local_opt: 
        penalty = np.sqrt(np.sum((p0 - opt1) ** 2)) * 1 
        return MSE + penalty 
    return MSE

# 
# Calibration 
#
def BCC_calibration_short(): 
    ''' Calibrates jump component of BCC97 model to market quotes. ''' 
    # first run with brute force 
    # (scan sensible regions) 
    global local_opt, opt1
    opt1 = 0.0 
    local_opt = True 
    opt1 = brute(BCC_error_function, 
                 ((0.0, 0.51, 0.1), # lambda 
                  (-0.5, -0.11, 0.1), # mu 
                  (0.0, 0.51, 0.25)), # delta 
                  finish=None)
    # second run with local, convex minimization 
    # (dig deeper where promising)
    opt2 = fmin(BCC_error_function, opt1, 
                xtol=0.0000001, ftol=0.0000001, 
                maxiter=550, maxfun=750) 
     
    return opt2

#==============================================================================
    
#==============================================================================
    
'''
Calibration of the full model
'''

def BCC_error_function_full(p0): 
    ''' Error function for parameter calibration in BCC97 model via 
    Lewis (2001) Fourier approach.
    
    Parameters 
    ========== 
    kappa_v: float 
        mean-reversion factor 
    theta_v: float 
        long-run mean of variance 
    sigma_v: float 
        volatility of variance 
    rho: float 
        correlation between variance and stock/index level 
    v0: float 
        initial, instantaneous variance 
    lamb: float 
        jump intensity 
    mu: float 
        expected jump size 
    delta: float 
        standard deviation of jump
        
    Returns 
    ======= 
    MSE: float 
        mean squared error 
    ''' 
    
    global i3, min_MSE3 
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = p0 
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or rho < -1.0 or rho > 1.0 or v0 < 0.0 or lamb < 0.0 or mu < -.6 or mu > 0.0 or delta < 0.0: 
        return 5000.0 
    if 2 * kappa_v * theta_v < sigma_v ** 2: 
        return 5000.0 
    se = [] 
    for option in options: 
        model_value = option.opt_BCC_value(kappa_v, theta_v, sigma_v, rho, v0,
                      lamb, mu, delta)
        se.append((model_value - option.close) ** 2)
    MSE = sum(se) / len(se) 
    min_MSE3 = min(min_MSE3, MSE)
    if i3 % 25 == 0: 
        print('%4d |' % i3, np.array(p0), '| %11.7f | %11.7f' % (MSE, min_MSE3))
    i3 += 1 
    return MSE

def BCC_calibration_full(): 
    ''' Calibrates complete BCC97 model to market quotes. ''' 
    # local, convex minimization for all parameters 
    opt = fmin(BCC_error_function_full, p1, 
               xtol=0.000001, ftol=0.000001, 
               maxiter=450, maxfun=650) 
    
    return opt

#%%
    
def bs(sigma,T,K,S0):
    d1=1/(sigma*T**(1/2))*(np.log(S0/K)+(r+sigma**2/2))
    d2=d1-sigma*T**(1/2)
    P=norm.cdf(-d2)*K*np.exp(-r*T)-norm.cdf(-d1)*S0
    return P

#%%
#==============================================================================
    
if __name__ == '__main__':

    if not os.path.exists('parameters_data/parameters_final.csv'):
        
        para_data=pd.DataFrame(None, columns=['kappa_v', 'theta_v', 'sigma_v', 'rho', 'v0', 'lamb', 'mu', 'delta'])
        
        data_list=os.listdir('option_data')
        
        for file in data_list:
            print(file)
            if 'csv' not in file:
                continue
    
            file="option_data/"+file
            global S0, r0, date

            data=pd.read_csv(file,index_col='DataDate')
              
            S0 = data["UnderlyingPrice"].iloc[1]
            r0 = 0.035 # Constant interest rate

            date = file[-18:-10] # Datadate
            
            _data = []
            for row, option in data.iterrows():
                opt = Option(option['OptionSymbol'], S0, option['Strike'], datetime.strptime(row, '%m/%d/%Y'), r0, datetime.strptime(option['Expiration'], '%m/%d/%Y'), 0.0088,
                             option['Mid'], option['Type'])
                _data.append(opt)
                    
            # 
            # Option Selection 
            # 
    
            options = []
            global TYPE
            TYPE = 'call' 
            for option in _data:
                if option.type == TYPE:
                    options.append(option)
        
        #==============================================================================
            
            print('Calibration of Stochastic Volatility...')
            opt = H93_calibration_full()   
            np.save('parameters/opt_sv_{0}_{1}'.format(date, TYPE), np.array(opt))  
            
            global kappa_v, theta_v, sigma_v, rho, v0
            kappa_v, theta_v, sigma_v, rho, v0 = np.load('parameters/opt_sv_{0}_{1}.npy'.format(date,TYPE)) # from H93 model calibration
            print('Calibration of Jump-Diffusion...')
            opt2 = BCC_calibration_short()     
            np.save('parameters/opt_jump_{0}_{1}'.format(date, TYPE), np.array(opt2)) 
        
            global lamb, mu, delta, p1
            lamb, mu, delta = np.load('parameters/opt_jump_{0}_{1}.npy'.format(date,TYPE)) # from BCC short model calibration
            p1 = [kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta]    
            print('Calibration of full model...')
            P = BCC_calibration_full() 
            
            para_data.loc[date]=P
            
        para_data.to_csv('parameters_data/parameters_final.csv')
    
#%%
#calculate delta
    r=0.035
    ep=0.01
    tp='call'

    call_list=os.listdir('Data_hedge')
    call_list.remove('.DS_Store')
    delta_BCC_data=pd.DataFrame(None, columns=call_list)
    delta_BS_data=pd.DataFrame(None, columns=call_list)
    profit_BCC_data=pd.DataFrame(None, columns=call_list) # without call
    profit_BS_data=pd.DataFrame(None, columns=call_list)
    profit_BCC_data2=pd.DataFrame(None, columns=call_list) # with call
    profit_BS_data2=pd.DataFrame(None, columns=call_list)
    call_BCC = pd.DataFrame(None, columns=call_list)
    call_BS = pd.DataFrame(None, columns=call_list)
    call_market = pd.DataFrame(None, columns=call_list)
    parameters=pd.read_csv('parameters_data/parameters_final.csv',index_col='DataDate')
    Date = pd.to_datetime(parameters.index, format='%Y%m%d')
    
    for file in call_list:
        
        data_hedge=pd.read_csv('Data_hedge/'+file,index_col='DataDate')
        for date in parameters.index:
            stock_price=data_hedge.loc[date,'Stock price']
            K=data_hedge.loc[date,'Strike']
            close=data_hedge.loc[date,'Mid']
            expire_date=str(data_hedge.loc[date,'Expiration'])
            expire_date=datetime(int(expire_date[0:4]),int(expire_date[4:6]),int(expire_date[6:8]))
            now=str(date)
            now=datetime(int(now[0:4]),int(now[4:6]),int(now[6:8]))
            para=parameters.loc[date,]
            opt_up=Option('aa',stock_price+ep,K,now,r,expire_date,para[4],close,tp)
            opt_down=Option('aa',stock_price-ep,K,now,r,expire_date,para[4],close,tp)
            price_up=opt_up.opt_BCC_value(para[0],para[1],para[2],para[3],para[4],para[5],para[6],para[7])
            price_down=opt_down.opt_BCC_value(para[0],para[1],para[2],para[3],para[4],para[5],para[6],para[7])
            delta_BCC=(price_up-price_down)/(2*ep)
            delta_BCC_data.loc[date,file]=delta_BCC
            
            op=Option('aaa',stock_price,K,now,r,expire_date,para[4],close,tp)
            delta_bs=op.opt_bs_delta()
            delta_BS_data.loc[date,file]=delta_bs
            
            call_BCC.loc[date,file] = op.opt_BCC_value(para[0],para[1],para[2],para[3],para[4],para[5],para[6],para[7])
            call_BS.loc[date,file] = op.opt_bs_value()
            call_market.loc[date,file] = op.close

#calculate profit
        for i,date in enumerate(parameters.index):
            stock_price=data_hedge.loc[date,'Stock price']
            if i==0:
                delta_BCC_pre=delta_BCC_data.loc[date,file]
                profit_BCC=delta_BCC_pre*stock_price
                delta_BS_pre=delta_BS_data.loc[date,file]
                profit_BS=delta_BS_pre*stock_price
                
                call_pre = call_market.loc[date,file]
                profit_BCC2 = delta_BCC_pre*stock_price - call_market.loc[date,file]
                profit_BS2 = delta_BS_pre*stock_price - call_market.loc[date,file]
                
            else:
                sub_BCC=delta_BCC_data.loc[date,file]-delta_BCC_pre
                sub_BS=delta_BS_data.loc[date,file]-delta_BS_pre
                delta_BCC_pre=delta_BCC_data.loc[date,file]
                delta_BS_pre=delta_BS_data.loc[date,file]
                profit_BCC=sub_BCC*stock_price
                profit_BS=sub_BS*stock_price
                
                sub_call = call_market.loc[date,file] - call_pre
                call_pre = call_market.loc[date,file]
                profit_BCC2 = sub_BCC*stock_price + sub_call
                profit_BS2 = sub_BS*stock_price + sub_call
                
            if i==len(parameters.index)-1:
                profit_BCC=-delta_BCC_pre*stock_price
                profit_BS=-delta_BS_pre*stock_price
                
                profit_BCC2 = -delta_BCC_pre * stock_price + call_market.loc[date,file]
                profit_BS2 = -delta_BS_pre * stock_price + call_market.loc[date,file]
                
    
            profit_BCC_data.loc[date,file]=profit_BCC
            profit_BS_data.loc[date,file]=profit_BS
            
            profit_BCC_data2.loc[date,file] = profit_BCC2
            profit_BS_data2.loc[date,file] = profit_BS2
            
            
    profit_BCC_data["average"] = profit_BCC_data.mean(axis=1)
    profit_BS_data["average"] = profit_BS_data.mean(axis=1)
    
    profit_BCC_data2["average"] = profit_BCC_data2.mean(axis=1)
    profit_BS_data2["average"] = profit_BS_data2.mean(axis=1)
    
    cum_profit_BCC = []
    cum_profit_BS = []
    for i in range(2, len(Date)):
        cum_profit_BCC.append(sum(profit_BCC_data2["average"][1:i]))
        cum_profit_BS.append(sum(profit_BS_data2["average"][1:i]))
        
    
    call_market["average"] = call_market.mean(axis=1)
    call_BCC["average"] = call_BCC.mean(axis=1)
    call_BS["average"] = call_BS.mean(axis=1)
    
# plot
    plt.figure()
    plt.plot(Date[1:-1],profit_BCC_data2["average"][1:-1], label = "BCC")
    plt.plot(Date[1:-1],profit_BS_data2["average"][1:-1], label = "BS")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.title("daily profit: BCC vs BS")
    
    plt.figure()
    plt.plot(Date[1:-1],cum_profit_BCC, label = "BCC")
    plt.plot(Date[1:-1],cum_profit_BS, label = "BS")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.title("cumulative profit: BCC vs BS")
    
    plt.figure()
    plt.plot(Date,call_market["average"], label = "market")
    plt.plot(Date,call_BCC["average"], label = "BCC")
    plt.plot(Date,call_BS["average"], label = "BS")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.title("call price: BCC vs BS vs market")
