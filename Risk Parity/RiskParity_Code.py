
# coding: utf-8

# ## risk parity function (equally-weighted risk contributions)

# In[1]:


def allocation_risk(weights, covariances):
    
    portfolio_risk = np.sqrt(np.dot(weights,np.dot(covariances,weights)))
    
    return portfolio_risk


# In[2]:


def assets_risk_contribution_to_allocation_risk(weights, covariances):

    portfolio_risk = allocation_risk(weights, covariances)
    assets_risk_contribution = weights*np.dot(covariances,weights)/ portfolio_risk

    return assets_risk_contribution


# In[3]:


def risk_budget_objective_error(weights, args):

    covariances = args[0]
    assets_risk_budget = args[1]
    weights = np.array(weights)
    portfolio_risk = allocation_risk(weights, covariances)
    assets_risk_contribution = assets_risk_contribution_to_allocation_risk(weights, covariances)
    assets_risk_target = np.multiply(portfolio_risk, assets_risk_budget)
    error = sum((assets_risk_contribution - assets_risk_target)**2)

    # It returns the calculated error
    return error



def max_drawdown(lt, windows):
    df_number = pd.DataFrame(lt)
    Roll_max = df_number.rolling(window = windows,min_periods = 1).max()
    
    Daily_drawdown = df_number/Roll_max-1
#  Max_Daily_Drawdown = Daily_drawdown.rolling(window = windows,min_periods =1).min()
    return Daily_drawdown

# In[4]:


def get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})
    constraints1 = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

    # Optimisation process in scipy
    optimize_result = minimize(fun=risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights


# In[5]:


def get_weights(covariances):

    # The desired contribution of each asset to the portfolio risk: we want all
    # asset to contribute equally
    assets_risk_budget = np.array([1 / covariances.shape[1]] * covariances.shape[1])
    # Initial weights: equally weighted
    init_weights = np.array([1 / covariances.shape[1]] * covariances.shape[1])
    # Optimisation process of weights
    weights = get_risk_parity_weights(covariances, assets_risk_budget, init_weights)
    # Convert the weights to a pandas Series
    weights = pd.Series(weights, index=covariances.columns, name='weight')
    # It returns the optimised weights
    return weights


# In[6]:


def get_IV_weights(covariances):
    diags=np.array(covariances).diagonal()
    inverse_diags=1/diags
    weights=inverse_diags/inverse_diags.sum()
    
    return weights


# In[7]:


def get_tangency_weights(covariances,train_data,rf):
    n = covariances.shape[0]
    v = np.ones(n)
    cov_inv = np.linalg.pinv(covariances)  
    tan_weights = np.dot(cov_inv,(np.mean(train_data)-rf)) / np.dot(np.dot(np.transpose(v),cov_inv),(np.mean(train_data)-rf))
    return tan_weights


# ## data preprocessing

# In[8]:


import pandas as pd
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
TOLERANCE = 1e-10

yf.pdr_override()

data = pdr.get_data_yahoo(
#    tickers =["SPY","BSV","LQD","IGIB","EMB","EEM","EFA","IVV","IXN","PDP"],
    tickers =["SPY","DGSIX","DFQTX","DFSHX","DFIEX","DFEOX","DFEQX","DFGBX","DFTEX","DFCEX","DIPSX","DWFIX"],
    #DGSIX:DFA Global Allocation 60/40 
    start = "2011-12-08",
    end="2018-12-31",
#    start = "2011-12-08",
#    end="2012-12-31",
#    start = "2012-09-27",
#    end="2013-12-31",
#    start = "2013-10-01",
#    end="2014-12-31",
#    start = "2011-12-08",
#    end="2014-12-31",
    as_panel = False,
    group_by = 'ticker',
    auto_adjust = True
    )

idxx = pd.IndexSlice
data=data.loc[idxx[:], idxx[:, 'Close']]
data.columns = data.columns.droplevel(1)
data=data.dropna(axis=0, how='any')


# ## rolling of return

# In[9]:

n_train = 63 # 3 month rolling data for training
n_test = 21 # 1 month rolling data for testing
#n_leverage = 2 # leverage
rf=0.03

ret=data.pct_change().dropna(axis=0,how='any')
ret_ewrisk=[]
ret_ivol=[]
ret_tangency = []
ret_spy=[]
ret_46=[]

ret_ewrisk_nl=[]
ret_ivol_nl=[]
ret_tangency_nl = []
ret_spy_nl=[]
ret_46_nl=[]


for i in np.arange(n_train,len(ret),n_test):
    
    ret_train = ret.iloc[(i - n_train):i,2:]
    cov = ret_train.cov()*252
    ew_risk = get_weights(cov)
    iv_risk = get_IV_weights(cov)
    tang_risk = get_tangency_weights(cov,ret_train,rf)
    
    ret_test = ret.iloc[i:(i + n_test),:]
    
    r_ewrisk = np.dot(ret_test.iloc[:,2:],ew_risk)
    r_ivol = np.dot(ret_test.iloc[:,2:],iv_risk)
    r_tangency = np.dot(ret_test.iloc[:,2:],tang_risk)
    r_spy = ret_test.iloc[:,0]
    r_46 = ret_test.iloc[:,1]
        
    ret_ewrisk.append(list(r_ewrisk))
    ret_ivol.append(list(r_ivol))
    ret_tangency.append(list(r_tangency))
    ret_spy.append(list(r_spy))
    ret_46.append(list(r_46))
    
    ret_ewrisk_nl.append(list(r_ewrisk))
    ret_ivol_nl.append(list(r_ivol))
    ret_tangency_nl.append(list(r_tangency))
    ret_spy_nl.append(list(r_spy))
    ret_46_nl.append(list(r_46))
    
    
ret_ewrisk = [item for sub in ret_ewrisk for item in sub]
ret_ivol = [item for sub in ret_ivol for item in sub]
ret_tangency = [item for sub in ret_tangency for item in sub]
ret_spy = [item for sub in ret_spy for item in sub]
ret_46 = [item for sub in ret_46 for item in sub]

ret_ewrisk_nl = [item for sub in ret_ewrisk_nl for item in sub]
ret_ivol_nl = [item for sub in ret_ivol_nl for item in sub]
ret_tangency_nl = [item for sub in ret_tangency_nl for item in sub]
ret_spy_nl = [item for sub in ret_spy_nl for item in sub]
ret_46_nl = [item for sub in ret_46_nl for item in sub]

# add leverage
ew=np.array(ret_ewrisk)
iv=np.array(ret_ivol)
ta=np.array(ret_tangency)
sp=np.array(ret_spy)
p46=np.array(ret_46)

lvg_ew=np.std(sp)/np.std(ew)
lvg_iv=np.std(sp)/np.std(iv)
lvg_ta=np.std(sp)/np.std(ta)
lvg_46=np.std(sp)/np.std(p46)

ret_ewrisk=ew*lvg_ew
ret_ivol=iv*lvg_iv
ret_tangency=ta*lvg_ta
ret_46=p46*lvg_46

# volatility plot after leverage
plt.plot(ret.index[n_train:],ret_ewrisk,label = 'risk parity portfolio')
plt.plot(ret.index[n_train:],ret_ivol,label = 'inverse vol portfolio')
plt.plot(ret.index[n_train:],ret_tangency,label = 'tangency portfolio')
plt.plot(ret.index[n_train:],ret_46,label = 'DFA Global 60/40 portfolio')
plt.plot(ret.index[n_train:],ret_spy,label = 'SPY')
plt.legend()
plt.xticks(rotation = 45, fontsize = 8)
plt.show()


# ## net value plot

# In[10]:

netvalue_ewrisk = [1]
netvalue_ivol = [1]
netvalue_tangency = [1]
netvalue_spy = [1]
netvalue_46 = [1]

for i in range(1,len(ret_spy)):
    netvalue_ewrisk.append(netvalue_ewrisk[-1]*(1+ret_ewrisk[i]))
    netvalue_ivol.append(netvalue_ivol[-1]*(1+ret_ivol[i]))
    netvalue_tangency.append(netvalue_tangency[-1]*(1+ret_tangency[i]))
    netvalue_spy.append(netvalue_spy[-1]*(1+ret_spy[i]))
    netvalue_46.append(netvalue_46[-1]*(1+ret_46[i]))
    

#plt.figure(figsize=(15,5))
plt.plot(ret.index[n_train:],netvalue_ewrisk,label = 'risk parity portfolio')
plt.plot(ret.index[n_train:],netvalue_ivol,label = 'inverse vol portfolio')
plt.plot(ret.index[n_train:],netvalue_tangency,label = 'tangency portfolio')
plt.plot(ret.index[n_train:],netvalue_46,label = 'DFA Global 60/40 portfolio')
plt.plot(ret.index[n_train:],netvalue_spy,label = 'SPY')
plt.legend()
plt.xticks(rotation = 45, fontsize = 8)
plt.show()


# ## Sharpe Ratio

# In[11]:


n_year = len(ret)/252
rf = 0.0

annret_ewrisk = np.mean(ret_ewrisk)*252
std_ewrisk = np.std(ret_ewrisk)*np.sqrt(252)
sharpe_ewrisk = (annret_ewrisk - rf)/std_ewrisk

annret_ivol = np.mean(ret_ivol)*252
std_ivol = np.std(ret_ivol)*np.sqrt(252)
sharpe_ivol = (annret_ivol - rf)/std_ivol

annret_tangency = np.mean(ret_tangency)*252
std_tangency = np.std(ret_tangency)*np.sqrt(252)
sharpe_tangency = (annret_tangency - rf)/std_tangency

annret_spy = np.mean(ret_spy)*252
std_spy = np.std(ret_spy)*np.sqrt(252)
sharpe_spy = (annret_spy - rf)/std_spy

annret_46 = np.mean(ret_46)*252
std_46 = np.std(ret_46)*np.sqrt(252)
sharpe_46 = (annret_46 - rf)/std_46

print("sharpe ratio of risk parity:",sharpe_ewrisk)
print("sharpe ratio of inverse volatility:",sharpe_ivol)
print("sharpe ratio of tangency:",sharpe_tangency)
print("sharpe ratio of spy:",sharpe_spy)
print("sharpe ratio of 40-60 portfolio:",sharpe_46)

sharpe = [sharpe_ewrisk,sharpe_ivol,sharpe_tangency,sharpe_spy,sharpe_46]

# In[12]:

## max drawdown (no leverage)

netvalue_ewrisk_nl = [1]
netvalue_ivol_nl = [1]
netvalue_tangency_nl = [1]
netvalue_spy_nl = [1]
netvalue_46_nl = [1]

for i in range(1,len(ret_spy_nl)):
    netvalue_ewrisk_nl.append(netvalue_ewrisk_nl[-1]*(1+ret_ewrisk_nl[i]))
    netvalue_ivol_nl.append(netvalue_ivol_nl[-1]*(1+ret_ivol_nl[i]))
    netvalue_tangency_nl.append(netvalue_tangency_nl[-1]*(1+ret_tangency_nl[i]))
    netvalue_spy_nl.append(netvalue_spy_nl[-1]*(1+ret_spy_nl[i]))
    netvalue_46_nl.append(netvalue_46_nl[-1]*(1+ret_46_nl[i]))


dw_ewrisk = max_drawdown(netvalue_ewrisk_nl,252)
dw_ivol = max_drawdown(netvalue_ivol,252)
dw_tangency = max_drawdown(netvalue_tangency,252)
dw_spy = max_drawdown(netvalue_spy_nl,252)
dw_46 = max_drawdown(netvalue_46_nl,252)
# Plot the results
plt.plot(ret.index[n_train:],dw_ewrisk,label = 'risk parity portfolio')
plt.plot(ret.index[n_train:],dw_ivol,label = 'inverse vol portfolio')
plt.plot(ret.index[n_train:],dw_tangency,label = 'tangency portfolio')
plt.plot(ret.index[n_train:],dw_spy,label = 'SPY')
plt.plot(ret.index[n_train:],dw_46,label = 'DFA Global 60/40 portfolio')
plt.legend()
plt.xticks(rotation = 45, fontsize = 8)
plt.show() 

print("the maximum drawdown of risk parity:",np.min(dw_ewrisk)[0])
print("the maximum drawdown of inverse volatility:",np.min(dw_ivol)[0])
print("the maximum drawdown of tangency:",np.min(dw_tangency)[0])
print("the maximum drawdown of spy:",np.min(dw_spy)[0])
print("the maximum drawdown of 60-40 porfolio:",np.min(dw_46)[0])

drawdown = [np.min(dw_ewrisk)[0],np.min(dw_ivol)[0],np.min(dw_tangency)[0],np.min(dw_spy)[0],np.min(dw_46)[0]]

# In[13]

## VAR (no leverage)

from scipy.stats import norm

alpha= 0.05


rp_mu1 = np.mean(ret_ewrisk_nl)
rp_std1 = np.std(ret_ewrisk_nl)
rp_VaR1 = norm.ppf(alpha, rp_mu1, rp_std1)

iv_mu1 = np.mean(ret_ivol_nl)
iv_std1 = np.std(ret_ivol_nl)
iv_VaR1 = norm.ppf(alpha, iv_mu1, iv_std1)


tan_mu1 = np.mean(ret_tangency_nl)
tan_std1 = np.std(ret_tangency_nl)
tan_VaR1 = norm.ppf(alpha,tan_mu1,tan_std1)


spy_mu1 = np.mean(ret_spy_nl)
spy_std1 = np.std(ret_spy_nl)
spy_VaR1 = norm.ppf(alpha, spy_mu1, spy_std1)

mu1_46 = np.mean(ret_46_nl)
std1_46 = np.std(ret_46_nl)
VaR1_46 = norm.ppf(alpha, mu1_46, std1_46)


print('VaR of risk parity:', rp_VaR1) 
print('VaR of inverse volatility:', iv_VaR1)     
print('VaR of tangency:', tan_VaR1) 
print('VaR of spy:', spy_VaR1) 
print('VaR of 60-40 portfolio:', VaR1_46) 

VAR = [rp_VaR1,iv_VaR1, tan_VaR1,spy_VaR1, VaR1_46 ]

# statistic table
stat_table = pd.DataFrame([sharpe,drawdown,VAR],columns=['risk parity', 'inverse volatility', 'tangency', 'SPY',"40-60 portfolio"], index = ['sharpe ratio','max drawdown','VAR'])
print(stat_table)

# In[14]

## Sharpe bar chart
date = ret.index[63:]
year_list = np.arange(2012,2019,1)
n_day = 252
sharpelist_ewrisk = []
sharpelist_46 = []
sharpelist_spy = []
annvol_spy = []


start = 0
j = 0
for year in year_list:
    
    while (date[j].year == year):
        j = j + 1
        if j == (len(date) - 1):
            break
        
    end = j
    
    sharpelist_ewrisk.append(np.mean(ret_ewrisk[start:end]) / np.std(ret_ewrisk[start:end]) * np.sqrt(n_day))
    sharpelist_46.append(np.mean(ret_46[start:end]) / np.std(ret_46[start:end]) * np.sqrt(n_day))
    sharpelist_spy.append(np.mean(ret_spy[start:end]) / np.std(ret_spy[start:end]) * np.sqrt(n_day))
    annvol_spy.append(np.std(ret_spy[start:end]) * np.sqrt(n_day))
    
    start = end

# plot 1
plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.bar(year_list-0.25,sharpelist_ewrisk,width = 0.25,label = "risk parity")
plt.bar(year_list,sharpelist_46,width = 0.25, label = "DFA 60-40 portfolio")
plt.bar(year_list+0.25,sharpelist_spy,width = 0.25, label = "SPY")
plt.xticks(rotation = 45, fontsize = 8)
plt.ylabel("Sharpe Ratio")
plt.title("Annualized Sharpe Ratio Plot")
plt.legend() 

plt.subplot(1,2,2)
plt.bar(year_list,annvol_spy)
plt.ylim([0.05,0.17])
plt.ylabel("Market Volatility")
plt.title("Annualized Market Volatility Plot")
plt.xticks(rotation = 45, fontsize = 8)

# plot 2
n = 3
n_list = np.arange(0,n,1)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.xticks(n_list,year_list[:n],rotation = 45, fontsize = 8)
plt.bar(n_list-0.25,sharpelist_ewrisk[:n],width = 0.25,label = "risk parity")
plt.bar(n_list,sharpelist_46[:n],width = 0.25, label = "DFA 60-40 portfolio")
plt.bar(n_list+0.25,sharpelist_spy[:n],width = 0.25, label = "SPY")
plt.ylabel("Sharpe Ratio")
plt.title("Annualized Sharpe Ratio Plot")
plt.legend() 

plt.subplot(1,2,2)
plt.xticks(n_list,year_list[:n],rotation = 45, fontsize = 8)
plt.bar(n_list,annvol_spy[:n],width = 0.4)
plt.ylim([0.1,0.17])
plt.ylabel("Market Volatility")
plt.title("Annualized Market Volatility Plot")

# In[15]

## Ann return bar chart

date = ret.index[63:]
year_list = np.arange(2012,2019,1)
n_day = 252
annret_ewrisk = []
annret_46 = []
annret_spy = []
annvol_spy = []


start = 0
j = 0
for year in year_list:
    
    while (date[j].year == year):
        j = j + 1
        if j == (len(date) - 1):
            break
        
    end = j
    
    annret_ewrisk.append(np.mean(ret_ewrisk[start:end]) * np.sqrt(n_day))
    annret_46.append(np.mean(ret_46[start:end]) * np.sqrt(n_day))
    annret_spy.append(np.mean(ret_spy[start:end]) * np.sqrt(n_day))
    annvol_spy.append(np.std(ret_spy[start:end]) * np.sqrt(n_day))
    
    start = end

plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.bar(year_list-0.25,annret_ewrisk,width = 0.25,label = "risk parity")
plt.bar(year_list,annret_46,width = 0.25, label = "DFA 60-40 portfolio")
plt.bar(year_list+0.25,annret_spy,width = 0.25, label = "SPY")
plt.xticks(rotation = 45, fontsize = 8)
plt.ylabel("Annualized Return")
plt.title("Annualized Return Plot")
plt.legend() 

plt.subplot(1,2,2)
plt.bar(year_list,annvol_spy)
plt.ylim([0.05,0.17])
plt.ylabel("Market Volatility")
plt.title("Annualized Market Volatility Plot")
plt.xticks(rotation = 45, fontsize = 8)

# plot 2
n = 3
n_list = np.arange(0,n,1)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.xticks(n_list,year_list[:n],rotation = 45, fontsize = 8)
plt.bar(n_list-0.25,annret_ewrisk[:n],width = 0.25,label = "risk parity")
plt.bar(n_list,annret_46[:n],width = 0.25, label = "DFA 60-40 portfolio")
plt.bar(n_list+0.25,annret_spy[:n],width = 0.25, label = "SPY")
plt.ylabel("Annualized Return")
plt.title("Annualized Return Plot")
plt.legend() 

plt.subplot(1,2,2)
plt.xticks(n_list,year_list[:n],rotation = 45, fontsize = 8)
plt.bar(n_list,annvol_spy[:n],width = 0.4)
plt.ylim([0.1,0.17])
plt.ylabel("Market Volatility")
plt.title("Annualized Market Volatility Plot")

# In[16]

# table 2

