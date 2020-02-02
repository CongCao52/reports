# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:45:56 2019

@author: DELL
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import pyfolio as pf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optimizer
import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline




#Create a Strategy #
class TestStrategy(bt.Strategy):

    params = (('vix_lowBound',30),\
              ('vix_upBound',20),\
              ('vixRatio10_buy',0.2),\
              ('vixRatio10_sell',0.4),\
              ('smaRatio_sell',1.015),\
              ('stop_loss',0.1),\
              ('trail',0.1)
              )

    def log(self, txt, dt=None,doprint=False):
        ''' Logging function fot this strategy'''
        if opt2 or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        super().__init__()
        self.broker.set_coc(True)

        self.dataclose={}
        self.dataopen={}
        self.datalow={}
        self.datahigh={}
        self.volume={}
        self.order={}
        self.trade_sig={}
        self.signal1={}
        self.signal2={} 
        self.signal3={} 
        self.close_sig={}
        self.w_portfolio=w_portfolio
        self.w_close=[0]*num
        self.option_ratio=option_ratio[opt]
        self.sma5 = bt.indicators.SMA(self.data, period=5)
        self.sma10 = bt.indicators.SMA(self.data, period=10) 
        self.sma20 = bt.indicators.SMA(self.data, period=20)
        self.sma30 = bt.indicators.SMA(self.data, period=30)
        self.sma60 = bt.indicators.SMA(self.data, period=60)         
        vix=Vix()
        vixRatio10=VixRatio10()      
        crossOver=bt.Or(bt.And(self.sma5/self.sma10<1,self.sma20/self.sma60<1,\
                         vixRatio10<=vixRatio10_buy[opt]),\
                         bt.And(self.sma5/self.sma10<1,self.sma20/self.sma60<1,\
                         vix<=self.p.vix_upBound))
        
        crossDown=bt.Or(vixRatio10>=self.p.vixRatio10_sell,\
                        self.sma5/self.sma10>self.p.smaRatio_sell)

        if opt6:
            percent=Percentile()        
            cover_call=percent<=0.2
            protect_put=percent>=0.8
            collar=bt.And(percent<0.8,percent>0.2)

        # Keep a reference to the "close" line in the data dataseries
        for i,data in enumerate(self.datas):
            self.dataclose[i] = self.datas[i].close
            self.dataopen[i]=self.datas[i].open
            self.datalow[i]=self.datas[i].low
            self.datahigh[i]=self.datas[i].high
            self.volume[i]=abs(self.datas[i].volume)
    
            # To keep track of pending orders and buy price/commission
            self.order[i] = None
            self.buyprice = None
            self.buycomm = None
            self.bar_executed=None
            self.trade_sig[i]=crossOver          
            self.close_sig[i]=crossDown
            
            if opt6:
                self.signal1[i]=cover_call
                self.signal2[i]=protect_put
                self.signal3[i]=collar  

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm= order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

## stop loss for each asset
#        if not self.getposition:  # we left the market

#            return
            
#        if not self.p.trail:

#            stop_price = order.executed.price * (1.0 - self.p.stop_loss)
#            self.close(exectype=bt.Order.Stop, price=stop_price)
                  
#        else:
#            self.close(exectype=bt.Order.StopTrail, trailamount=self.p.trail)        
        
        for i,data in enumerate(self.datas):
            self.order[i] = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        
        # Simply log the closing price of the series from the reference
#        self.log('Close, %.2f' % self.dataclose[0])
        for i, data in enumerate(self.datas):
            data_name = data._name
            pos = self.getposition(data).size
            self.log('{} Position {}'.format(data_name, pos))

        # Check if an order is pending ... if yes, we cannot send a 2nd one
            if self.order[i]:
                return
            
            
            else:
                if rebalance=="no rebalance":
                    if not self.getposition(data):
                        self.order[i]=self.order_target_percent(data=self.datas[i],\
                                                  target=self.w_portfolio[i])                    
                
                
                elif rebalance=="rebalance":
                    
                    if opt6:

                        if not self.getposition(data) and self.trade_sig[i]:
                            
                            if i<num-2:
                                self.order[i]=self.order_target_percent(data=self.datas[i],\
                                                          target=self.w_portfolio[i])
                            
                            else:
                                
                                if self.signal1[i]:
                                    
                                    size=[0]*(num-2)+[int(self.volume[num-2]*self.option_ratio),0]
                                    self.order[i]=self.order_target_size(data=self.datas[i],\
                                                      target=size[i])
                                    
                                
                                elif self.signal2[i]:
                                    
                                    size=[0]*(num-2)+[0,int(self.volume[num-1]*self.option_ratio)]
                                    self.order[i]=self.order_target_size(data=self.datas[i],\
                                                      target=size[i])                           
                        
                                else:
                                    
                                    size=[0]*(num-2)+[int(self.volume[num-2]*self.option_ratio),int(self.volume[num-1]*0.1)]
                                    self.order[i]=self.order_target_size(data=self.datas[i],\
                                                      target=size[i])
    
    
                        if i>=num-2 and self.getposition(self.datas[i]):                                               
                            
                            if self.data.datetime.date().weekday() == 2 and \
                            str(self.data.datetime.date())[8:]>='25':
                                        
                                self.order[i]=self.order_target_size(data=self.datas[i],\
                                          target=self.w_close[i])
                        
         
                        if i>=num-2 and not self.getposition(self.datas[num-1]) and\
                        not self.getposition(self.datas[num-2]) and self.getposition(self.datas[0]):
                            
                            if self.signal1[i]:
                                    
                                size=[0]*(num-2)+[int(self.volume[num-2]*self.option_ratio),0]
                                self.order[i]=self.order_target_size(data=self.datas[i],\
                                                  target=size[i])
                                
                            
                            elif self.signal2[i]:
                                
                                size=[0]*(num-2)+[0,int(self.volume[num-1]*self.option_ratio)]
                                self.order[i]=self.order_target_size(data=self.datas[i],\
                                                  target=size[i])                           
                    
                            else:
                                
                                size=[0]*(num-2)+[int(self.volume[num-2]*self.option_ratio),int(self.volume[num-1]*0.1)]
                                self.order[i]=self.order_target_size(data=self.datas[i],\
                                                  target=size[i])
                    else:

                        if not self.getposition(data) and self.trade_sig[i]:
                        
                            self.order[i]=self.order_target_percent(data=self.datas[i],\
                                                      target=self.w_portfolio[i])                        
                        
                    if self.getposition(data) and self.close_sig[i]:
                        self.order[i]=self.close(data=self.datas[i])
                


#           last day close position                       
                if len(self.datas[i]) == self.datas[i].buflen()-1:
                    self.order[i]=self.close(data=self.datas[i])
            
            

                    
    def stop(self):                      
        self.log('(vixRatio_sell %.2f) Ending Value %.2f'%\
                 (self.params.vixRatio10_sell,\
                  self.broker.getvalue()),doprint=True)


if __name__ == '__main__':
    
    '''Option setting'''
    opt=1# CN=0,US=1
    opt2=1# optimization=0,backtest=1
    opt3=1# portfolio with index=0,portfolio with stock=1
    opt4=1#no rebalance=0,rebalance=1
    opt5=0# inverse ETF weight option: 0.1,0.2,0.3,0.4,0.5
    opt6=1#no option=0,with option=1
    
    read_path='data/'
    save_path='parameter result'
    name=['data_CN','data_US']
    name2=['stock_CN','stock_US'] 
    name8=['data_optCN','data_optUS']

    indexName_CN=['SSE50','SSE50 Short']
    stockName_CN=['中国平安','贵州茅台','招商银行','伊利股份',\
         '恒瑞医药','保利地产','中信证券','海螺水泥','三一重工','上汽集团']
    indexName_US=['SPY500','SPY500 Short']
    stockName_US=['MicroSoft','Apple','Berkshire Hathaway','JP Morgan',\
                  'Johnson','Exxon Mobil','AT&T','Procter & Gamble',\
                  'Disney','Coca Cola']
    
    optionName_CN=['SSE50 call','SSE50 put']
    optionName_US=['S&P500 call','S&P500 put']
    
    name3=[indexName_CN,indexName_US]
    name4=[stockName_CN,stockName_US]
    name5=[name3,name4]
    name9=[optionName_CN,optionName_US]

    name6=['data_IVIX','data_VIX']
    name7=['percentile_CN','percentile_US']

    index=pd.read_excel(read_path+name[opt]+'.xlsx')['close']
    vix=pd.read_excel(read_path+name6[opt]+'.xlsx')['close']
    percent=pd.read_excel(read_path+name7[opt]+'.xlsx')['percentage']
    
    startDate=['2015-03-01','2016-05-01']#for CN and US
    
    result_name=["performance_CN.csv","performance_US.csv"]
    
    
    commission_ratio=0.0003
    cash_ratio=0.005## for commission and option payment
    cash=500000
    count=0##stock count
    optimize_count=0## optimization count
    days=252#trading day
    mday=22#trading day a month
    num_list=[4,13]##asset number list
    num=num_list[opt3]##asset number: 2 for portfolio with index,11 for portfolio with stock
    risk_averse=2#risk preference
    vixRatio10_buy=[-0.2,0.2]
    option_ratio=[0.05,0.05]
    

    situations=["optimization","backtest"]
    rebalance_list=["no rebalance","rebalance"]
    situation=situations[opt2]
    rebalance=rebalance_list[opt4]
    performance=pd.DataFrame(columns=["n","Profit","Sharpe Ratio",\
                                          "Max Drawdown","Trade Num"])
    stock=pd.read_excel(read_path+name2[opt]+'.xlsx')
    stock=stock.iloc[:,range(3,len(stock.columns),5)]
    stock.columns=name4[opt]
    histRtn=np.log(stock/stock.shift(1))
    histRtn=histRtn.fillna(0)
    annRtn=histRtn.mean()*days
    cov=histRtn.cov()*days
    
    optimizer_stock=optimizer.optimizer()
    w=optimizer_stock.optimize(annRtn,risk_averse,cov)
    w_asset=[0.1,0.2,0.3,0.4,0.5]
    w_stock=[w_asset[opt5]*(1-cash_ratio)]+((1-w_asset[opt5])*w*(1-cash_ratio)).tolist()
    w_index=[w_asset[opt5]*(1-cash_ratio),(1-w_asset[opt5])*(1-cash_ratio)]
    w_list=[w_index,w_stock]
    w_portfolio=w_list[opt3]
    
               

    class Vix(bt.Indicator):
        
        lines=('vix',)
        
        def next(self):
            self.lines.vix[0]=vix[0]
            
        def once(self, start, end):
           vix_array = self.lines.vix.array
    
           for i in range(start, end):
               vix_array[i] = vix[i]
    
    class VixRatio10(bt.Indicator):
        
        lines=('vixRatio10',)
        
        def next(self):
            self.lines.vixRatio10[0]=0
            
        def once(self, start, end):
           vixRatio10_array = self.lines.vixRatio10.array
    
           for i in range(start, end):
               vixRatio10_array[i] = vix[i]/vix[i-10]-1
               

    class Percentile(bt.Indicator):
        
        lines=('percent',)
        
        def next(self):
            self.lines.percent[0]=percent[0]
            
        def once(self, start, end):
           percent_array = self.lines.percent.array
    
           for i in range(start, end):
               percent_array[i] = percent[i]
               
    
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    # Set the commission - 0.03% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=commission_ratio)
    # Add a FixedSize sizer according to the stake
#    cerebro.addsizer(bt.sizers.FixedSize, stake=1)
    
    # Add a strategy
    if situation=="backtest":
        cerebro.addstrategy(TestStrategy)  
    elif situation=="optimization":
        cerebro.optstrategy(TestStrategy,vixRatio10_buy=[x/10 for x in range(-5,5,1)])

    
    # Add the Data Feed to Cerebro    
    
    rawData=pd.read_excel(read_path+name[opt]+'.xlsx')
    rawData=rawData.loc[startDate[opt]:'2019',:]
    rawData['open.1']=rawData['close.1']
    rawData['high.1']=rawData['close.1']
    rawData['low.1']=rawData['close.1']
    rawData=rawData.dropna()
    benchmark=(rawData['close'].diff()/rawData['close'].shift(1)).fillna(0)
    benchmark=benchmark.rename('Benchmark')
    benchmark.index=benchmark.index.tz_localize('UTC')
    
    optData=pd.read_excel(read_path+name8[opt]+'.xlsx')
    optData.loc[startDate[opt]:'2019',:]
    optData.index=pd.to_datetime(optData.index)
    optData=optData.dropna()
    
    tempData=rawData.iloc[:,5:10]
    tempData.columns=['open','high','low','close','volume']
    data0=bt.feeds.PandasData(dataname=tempData,openinterest=None)
    cerebro.adddata(data0,name=name3[opt][1])
    
    data_range=[range(0,5,5),range(10,len(rawData.columns),5)]
    #10,len(rawData.columns),5 for stocks; 0,5,5 for index
    
    for i in data_range[opt3]:
        tempData=rawData.iloc[:,i:i+5]
        tempData.columns=['open','high','low','close','volume']        
        data=bt.feeds.PandasData(dataname=tempData,openinterest=None)
        data.plotinfo.plotmaster=data0
        cerebro.adddata(data,name=name5[opt3][opt][count])
        count+=1
        
    count=0
    
    if opt6:
        for i in range(0,len(optData.columns),5):
            tempData=np.power(-1,count+1)*optData.iloc[:,i:i+5]
            tempData.columns=['open','high','low','close','volume']       
            data=bt.feeds.PandasData(dataname=tempData,openinterest=None)
            data.plotinfo.plotmaster=data0
            cerebro.adddata(data,name=name9[opt][count])
            count+=1
    
    cerebro.addanalyzer(bt.analyzers.PyFolio)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='Sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='Drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='Trade')

    if situation=="backtest":
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    results=cerebro.run(maxcpus=1)
    
    if situation=="backtest":    
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())  
    
#    cerebro.plot(volume=False)

    if situation=="backtest":


        pyfoliozer = results[0].analyzers.getbyname('pyfolio')
        portfolio_rets, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    #utc time
        portfolio_rets.index = portfolio_rets.index.tz_convert('UTC')
        positions.index = positions.index.tz_convert('UTC')
        transactions.index = transactions.index.tz_convert('UTC')
    
    # pyfolio show
    
        pf.create_full_tear_sheet(
            returns=portfolio_rets,
            positions=positions,
            transactions=transactions,
    #        gross_lev=gross_lev,
            live_start_date=None,  # This date is sample specific
            cone_std=(1.0, 1.5, 2.0),
            benchmark_rets=benchmark,
            bootstrap=False,
            turnover_denom='AGB',
            header_rows=None)

    elif situation=="optimization":

        result =  [x[0] for x in results]
        
        for res in result:
            performance.loc[optimize_count,"n"]=res.params.vix_lowBound
            performance.loc[optimize_count,"Profit"]=round(res.analyzers.Trade.\
                           get_analysis()["pnl"]["gross"]['total'],2)
            performance.loc[optimize_count,"Sharpe Ratio"]=round(res.analyzers.\
                           Sharpe.get_analysis()['sharperatio'],2)
            performance.loc[optimize_count,"Max Drawdown"]=round(res.analyzers.Drawdown.\
                           get_analysis().max.drawdown,2)
            performance.loc[optimize_count,"Trade Num"]=round(res.analyzers.Trade.\
                           get_analysis()['total']['total'],2)      
            optimize_count+=1
        
        performance.to_csv(save_path+result_name[opt])
    
