import dash
import base64
import io
import numpy as np
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import pandas as pd
import pandas_datareader as pdr
import datetime
from datetime import date
import requests
import math
import plotly.graph_objects as go
import plotly
from bs4 import BeautifulSoup
import yfinance as yf
import pandas_ta as pta


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        return pd.read_excel(io.BytesIO(decoded))

    

def Model_Display(total_value, reason, rows):
    df = pd.DataFrame(rows)
    df['Dollar Allocation'] = df['Weights']*total_value
    df1 = pd.DataFrame()
    amount_of_shares = []
    count = 0
    for h in df['Holdings']:
        data = yf.download(h,start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
        df2 = pd.DataFrame(data)
        df1[h] = df2['Close']
        df1[h][0] = df2['Close'][1]
        amount_of_shares.append(df['Dollar Allocation'][count]/df1[h][0])
        count = count + 1
    market1 = yf.download("VOO",start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
    united1 = total_value/market1['Close'][0]
    market2 = yf.download("QQQ",start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
    united2 = total_value/market2['Close'][0]
    market3 = yf.download("IOZ.AX",start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
    united3 = total_value/market3['Close'][0]
    market4 = yf.download("VTS.AX",start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
    SPXU = yf.download("SPXU",start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
    SQQQ = yf.download("SQQQ",start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
    TQQQ = yf.download("TQQQ",start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
    UPRO = yf.download("UPRO",start =(date.today() - datetime.timedelta(days=2*365)), end = date.today())
    united4 = total_value/market4['Close'][0]
    market_portfolio1 = []
    market_portfolio2 = []
    market_portfolio3 = []
    market_portfolio4 = []
    df['Units'] = amount_of_shares
    count1 = 0
    count2 = 0
    portfolio_value = []
    quick_sum = 0
    while count1 < len(df1[df['Holdings'][0]]):
        market_portfolio1.append(united1*market1['Close'][count1])
        market_portfolio2.append(united2*market2['Close'][count1])
        market_portfolio3.append(united3*market3['Close'][count1])
        market_portfolio4.append(united4*market4['Close'][count1])
        count2 = 0
        quick_sum = 0
        while count2 < len(df['Holdings']):
            quick_sum = df['Units'][count2]*df1[df['Holdings'][count2]][count1] + quick_sum
            count2 = count2 + 1
        portfolio_value.append(round(quick_sum,2))
        count1 = count1 + 1
    df1['Portfolio'] = portfolio_value      
    df1['Market1'] = market_portfolio1
    df1['Market2'] = market_portfolio2
    df1['Market3'] = market_portfolio3
    df1['Market4'] = market_portfolio4
    df1['26 EMA'] = df1.ewm(span = 26, min_periods = 26).mean()['Portfolio']
    df1['12 EMA'] = df1.ewm(span = 12, min_periods = 12).mean()['Portfolio']
    df1['MACD'] = df1['12 EMA'] - df1['26 EMA']
    df1['MACD MEAN'] = df1['MACD'].mean()
    df1['Signal Line'] = df1.ewm(span = 9, min_periods = 9).mean()['MACD']
    df1['RSI'] = pta.rsi(df1['Portfolio'], length = 14)
    df1['RSI MEAN'] = df1['RSI'].mean()
    portfolio_advanced = []
    RSI_protect_price = []
    MACD_protect_price = []
    RSI_protect_date = []
    MACD_protect_date = []
    RSI_enhance_price = []
    MACD_enhance_price = []
    RSI_enhance_date = []
    MACD_enhance_date = []
    P_enhance_price = []
    P_enhance_date = []
    P_protect_price = []
    P_protect_date = []
    x = 4
    short_time = 0
    long_time = 0
    Enhanced_Portfolio_Value = [df1['Portfolio'][0],df1['Portfolio'][1],df1['Portfolio'][2],df1['Portfolio'][3]]
    Order_Value = 0
    Order_Status = ""
    SQQQ_Units = 0
    SPXU_Units = 0
    TQQQ_Units = 0
    UPRO_Units = 0
    Profit_Taken = 0
    neutral_time = 0
    while x < len(df1[df['Holdings'][0]]):
        if df1['MACD'][x]>df1['MACD MEAN'][x]:
            if df1['MACD'][x-1] > df1['Signal Line'][x-1] and df1['MACD'][x-2] > df1['Signal Line'][x-2] and df1['MACD'][x-3] > df1['Signal Line'][x-3]:
                if df1['MACD'][x-2]>df1['MACD'][x-3] and df1['MACD'][x-3]>df1['MACD'][x-4]:
                    if df1['MACD'][x] < df1['MACD'][x-1] and df1['MACD'][x-1] < df1['MACD'][x-2]:
                        if df1['RSI'][x-1] > df1['RSI MEAN'][x-1]:
                            if short_time == 0:
                                print("1",df1.index.date[x])
                                Profit_Taken = Profit_Taken + TQQQ_Units*TQQQ['Close'][x]+UPRO_Units*UPRO['Close'][x] - Order_Value
                                TQQQ_Units = 0
                                UPRO_Units = 0
                                Order_Value = 0.2*df1['Portfolio'][x]
                                Order_Status = "SHORT"
                                SQQQ_Units = (Order_Value/2)/SQQQ['Close'][x]
                                SPXU_Units = (Order_Value/2)/SPXU['Close'][x]
                                RSI_protect_price.append(df1['RSI'][x])
                                MACD_protect_price.append(df1['MACD'][x])
                                RSI_protect_date.append(df1.index.date[x])
                                MACD_protect_date.append(df1.index.date[x])
                                P_protect_price.append(df1['Portfolio'][x])
                                P_protect_date.append(df1.index.date[x])
                                short_time = x
                                long_time = 0
                                neutral_time = 0
        else:
            if df1['MACD'][x] < df1['Signal Line'][x] and df1['MACD'][x-1] < df1['Signal Line'][x-1] and df1['MACD'][x-2] < df1['Signal Line'][x-2]:
                if df1['MACD'][x-1]<df1['MACD'][x-2] and df1['MACD'][x-2]<df1['MACD'][x-3] and df1['MACD'][x-3]<df1['MACD'][x-4]:
                    if df1['MACD'][x] > df1['MACD'][x-1]:
                        if df1['RSI'][x] < df1['RSI MEAN'][x]:
                            if long_time == 0:
                                print("2",df1.index.date[x])
                                Profit_Taken = Profit_Taken + SQQQ_Units*SQQQ['Close'][x]+SPXU_Units*SPXU['Close'][x] - Order_Value
                                SQQQ_Units = 0
                                SPXU_Units = 0
                                Order_Value = 0.2*df1['Portfolio'][x]
                                Order_Status = "LONG"
                                TQQQ_Units = (Order_Value/2)/TQQQ['Close'][x]
                                UPRO_Units = (Order_Value/2)/UPRO['Close'][x]
                                short_time = 0
                                long_time = x
                                RSI_enhance_price.append(df1['RSI'][x])
                                MACD_enhance_price.append(df1['MACD'][x])
                                RSI_enhance_date.append(df1.index.date[x])
                                MACD_enhance_date.append(df1.index.date[x])
                                P_enhance_price.append(df1['Portfolio'][x])
                                P_enhance_date.append(df1.index.date[x])
                                neutral_time = 0
        if df1['MACD'][x-1]<df1['MACD'][x-2]:
            if df1['MACD'][x]>df1['MACD'][x-1]:
                if SQQQ_Units > 0:
                    print("3",df1.index.date[x])
                    Profit_Taken = Profit_Taken + SQQQ_Units*SQQQ['Close'][x]+SPXU_Units*SPXU['Close'][x] - Order_Value
                    SQQQ_Units = 0
                    SPXU_Units = 0
                    Order_Value = 0
                    short_time = 0
                    SQQQ_Units = 0
                    SPXU_Units = 0
                    Order_Status = "NEUTRAL"
                    neutral_time = x
        if df1['MACD'][x-1]>df1['MACD'][x-2]:
            if df1['MACD'][x]<df1['MACD'][x-1]:
                if TQQQ_Units > 0:
                    print("4",df1.index.date[x])
                    Profit_Taken = Profit_Taken + TQQQ_Units*TQQQ['Close'][x]+UPRO_Units*UPRO['Close'][x] - Order_Value
                    TQQQ_Units = 0
                    UPRO_Units = 0
                    Order_Value = 0
                    long_time = 0
                    TQQQ_Units = 0
                    UPRO_Units = 0
                    Order_Status = "NEUTRAL"
                    neutral_time = x
        if neutral_time > 0:
            if df1['MACD'][x] < MACD_enhance_price[len(MACD_enhance_price)-1]:
                if x - 1 < neutral_time:
                    print("5",df1.index.date[x])
                    Profit_Taken = Profit_Taken + TQQQ_Units*TQQQ['Close'][x]+UPRO_Units*UPRO['Close'][x] - Order_Value
                    TQQQ_Units = 0
                    UPRO_Units = 0
                    Order_Value = 0.2*df1['Portfolio'][x]
                    Order_Status = "SHORT"
                    SQQQ_Units = (Order_Value/2)/SQQQ['Close'][x]
                    SPXU_Units = (Order_Value/2)/SPXU['Close'][x]
                    RSI_protect_price.append(df1['RSI'][x])
                    MACD_protect_price.append(df1['MACD'][x])
                    RSI_protect_date.append(df1.index.date[x])
                    MACD_protect_date.append(df1.index.date[x])
                    P_protect_price.append(df1['Portfolio'][x])
                    P_protect_date.append(df1.index.date[x])
                    short_time = x
                    long_time = 0
                    neutral_time = 0
            if df1['MACD'][x] > MACD_protect_price[len(MACD_protect_price)-1]:
                if x - 1 < neutral_time:
                    print("6",df1.index.date[x])
                    Profit_Taken = Profit_Taken + SQQQ_Units*SQQQ['Close'][x]+SPXU_Units*SPXU['Close'][x] - Order_Value
                    SQQQ_Units = 0
                    SPXU_Units = 0
                    Order_Value = 0.2*df1['Portfolio'][x]
                    Order_Status = "LONG"
                    TQQQ_Units = (Order_Value/2)/TQQQ['Close'][x]
                    UPRO_Units = (Order_Value/2)/UPRO['Close'][x]
                    short_time = 0
                    long_time = x
                    neutral_time = 0
                    RSI_enhance_price.append(df1['RSI'][x])
                    MACD_enhance_price.append(df1['MACD'][x])
                    RSI_enhance_date.append(df1.index.date[x])
                    MACD_enhance_date.append(df1.index.date[x])
                    P_enhance_price.append(df1['Portfolio'][x])
                    P_enhance_date.append(df1.index.date[x])
        if Order_Value > 0:
            if Order_Status == "SHORT":
                Enhanced_Portfolio_Value.append(df1['Portfolio'][x]+SQQQ_Units*SQQQ['Close'][x]+SPXU_Units*SPXU['Close'][x] - Order_Value + Profit_Taken)
            if Order_Status == "LONG":
                Enhanced_Portfolio_Value.append(df1['Portfolio'][x]+TQQQ_Units*TQQQ['Close'][x]+UPRO_Units*UPRO['Close'][x] - Order_Value + Profit_Taken)
        else:
            Enhanced_Portfolio_Value.append(df1['Portfolio'][x] + Profit_Taken)
        print(Profit_Taken)
        x = x + 1
    df1['Enhanced Portfolio'] = Enhanced_Portfolio_Value
    df1 = df1.bfill(axis ='rows')
    if reason == 'figure':
        fig = go.Figure()
        king = ('Portfolio Performance of Lump Sum Invested 2 Years Ago')
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Portfolio'], mode = 'lines', name = 'Portfolio',marker=dict(size=1, color="blue")))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Enhanced Portfolio'], mode = 'lines', name = 'Enhanced Portfolio',marker=dict(size=1, color="blue")))
        df4 = pd.DataFrame(data = {'Dates1':P_protect_date,'SellPrice1':P_protect_price})
        fig.add_trace(go.Scatter(x=df4['Dates1'],y=df4['SellPrice1'], mode = 'markers',marker=dict(size=12, color="Orange"),showlegend=False))
        df5 = pd.DataFrame(data = {'Dates1':P_enhance_date,'BuyPrice1':P_enhance_price})
        fig.add_trace(go.Scatter(x=df5['Dates1'],y=df5['BuyPrice1'], mode = 'markers',marker=dict(size=12, color="Green"),showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Market1'], mode = 'lines', name = 'S&P 500 Benchmark',marker=dict(size=1, color="red")))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Market2'], mode = 'lines', name = 'Nasdaq Benchmark',marker=dict(size=1, color="green")))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Market3'], mode = 'lines', name = 'ASX 200 Benchmark',marker=dict(size=1, color="purple")))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Market4'], mode = 'lines', name = 'US Total Market Benchmark',marker=dict(size=1, color="orange")))
        fig.update_layout(title=king,xaxis_title="Time",yaxis_title="Portfolio Value", width=1100, height = 700)
        fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
        return fig
    if reason == 'MACD':
        fig = go.Figure()
        king = ('MACD Chart')
        df4 = pd.DataFrame(data = {'Dates1':MACD_protect_date,'SellPrice1':MACD_protect_price})
        fig.add_trace(go.Scatter(x=df4['Dates1'],y=df4['SellPrice1'], mode = 'markers',marker=dict(size=12, color="Orange"),showlegend=False))
        df5 = pd.DataFrame(data = {'Dates1':MACD_enhance_date,'BuyPrice1':MACD_enhance_price})
        fig.add_trace(go.Scatter(x=df5['Dates1'],y=df5['BuyPrice1'], mode = 'markers',marker=dict(size=12, color="Green"),showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['MACD'], mode = 'lines',marker=dict(size=1, color="red"),showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Signal Line'], mode = 'lines',marker=dict(size=1, color="green"),showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['MACD MEAN'], mode = 'lines',marker=dict(size=1, color="yellow"),showlegend=False))
        fig.update_layout(title=king,xaxis_title="Time",yaxis_title="MACD Value", width=700, height = 500)
        return fig
    if reason == 'RSI':
        df1['buy']= 20
        df1['sell'] = 80
        fig = go.Figure()
        king = ('RSI Chart')
        df4 = pd.DataFrame(data = {'Dates1':RSI_protect_date,'SellPrice1':RSI_protect_price})
        fig.add_trace(go.Scatter(x=df4['Dates1'],y=df4['SellPrice1'], mode = 'markers',marker=dict(size=12, color="Orange"),showlegend=False))
        df5 = pd.DataFrame(data = {'Dates1':RSI_enhance_date,'BuyPrice1':RSI_enhance_price})
        fig.add_trace(go.Scatter(x=df5['Dates1'],y=df5['BuyPrice1'], mode = 'markers',marker=dict(size=12, color="Green"),showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['RSI'], mode = 'lines',marker=dict(size=1, color="blue"),showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['buy'], mode = 'lines',marker=dict(size=1, color="green"),showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['sell'], mode = 'lines',marker=dict(size=1, color="red"),showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['RSI MEAN'], mode = 'lines',marker=dict(size=1, color="yellow"),showlegend=False))
        fig.update_layout(title=king,xaxis_title="Time",yaxis_title="RSI Value", width=700, height = 500)
        return fig
    df4 = pd.DataFrame()
    portfolio = []
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    x = 1
    p_ret = [0]
    m1_ret = [0]
    m2_ret = [0]
    m3_ret = [0]
    m4_ret = [0]
    while x < len(df1['Portfolio']):
        p_ret.append((df1['Portfolio'][x]-df1['Portfolio'][x-1])/df1['Portfolio'][x-1])
        m1_ret.append((df1['Market1'][x]-df1['Market1'][x-1])/df1['Market1'][x-1])
        m2_ret.append((df1['Market2'][x]-df1['Market2'][x-1])/df1['Market2'][x-1])
        m3_ret.append((df1['Market3'][x]-df1['Market3'][x-1])/df1['Market3'][x-1])
        m4_ret.append((df1['Market4'][x]-df1['Market4'][x-1])/df1['Market4'][x-1])
        x = x + 1
    df1['P Ret'] = p_ret
    df1['M1 Ret'] = m1_ret
    df1['M2 Ret'] = m2_ret
    df1['M3 Ret'] = m3_ret
    df1['M4 Ret'] = m4_ret
    if reason == 'market':
        measures = ['Best Month','Worst Month','Downside Deviation','Worst 1 Day', 'Maximum Drawdown']
        df1['P Month'] = df1.rolling(window=21).sum()['P Ret']
        df1['M1 Month'] = df1.rolling(window=21).sum()['M1 Ret']
        df1['M2 Month'] = df1.rolling(window=21).sum()['M2 Ret']
        df1['M3 Month'] = df1.rolling(window=21).sum()['M3 Ret']
        df1['M4 Month'] = df1.rolling(window=21).sum()['M4 Ret']
        portfolio.append(round(max(df1['P Month'][22:len(df1['P Month'])-1]),3))
        m1.append(round(max(df1['M1 Month'][22:len(df1['P Month'])-1]),3))
        m2.append(round(max(df1['M2 Month'][22:len(df1['P Month'])-1]),3))
        m3.append(round(max(df1['M3 Month'][22:len(df1['P Month'])-1]),3))
        m4.append(round(max(df1['M4 Month'][22:len(df1['P Month'])-1]),3))
        portfolio.append(round(min(df1['P Month'][22:len(df1['P Month'])-1]),3))
        m1.append(round(min(df1['M1 Month'][22:len(df1['P Month'])-1]),3))
        m2.append(round(min(df1['M2 Month'][22:len(df1['P Month'])-1]),3))
        m3.append(round(min(df1['M3 Month'][22:len(df1['P Month'])-1]),3))
        m4.append(round(min(df1['M4 Month'][22:len(df1['P Month'])-1]),3))
        dfp = df1.loc[df1['P Ret']<0]
        portfolio.append(round(dfp['P Ret'].std(),3))
        dfm1 = df1.loc[df1['M1 Ret']<0]
        m1.append(round(dfp['M1 Ret'].std(),3))
        dfm2 = df1.loc[df1['M2 Ret']<0]
        m2.append(round(dfp['M2 Ret'].std(),3))
        dfm3 = df1.loc[df1['M3 Ret']<0]
        m3.append(round(dfp['M3 Ret'].std(),3))
        dfm4 = df1.loc[df1['M4 Ret']<0]
        m4.append(round(dfp['M4 Ret'].std(),3))
        portfolio.append(round(min(df1['P Ret']),3))
        m1.append(round(min(df1['M1 Ret']),3))
        m2.append(round(min(df1['M2 Ret']),3))
        m3.append(round(min(df1['M3 Ret']),3))
        m4.append(round(min(df1['M4 Ret']),3))
        x = 1
        DD_P = [0]
        DD_M1 = [0]
        DD_M2 = [0]
        DD_M3 = [0]
        DD_M4 = [0]
        while x < len(df1['P Ret']):
            MaxP = max(df1['Portfolio'][0:x])
            MaxM1 = max(df1['Market1'][0:x])
            MaxM2 = max(df1['Market2'][0:x])
            MaxM3 = max(df1['Market3'][0:x])
            MaxM4 = max(df1['Market4'][0:x])
            DD_P.append((df1['Portfolio'][x]-MaxP)/MaxP)
            DD_M1.append((df1['Market1'][x]-MaxM1)/MaxM1)
            DD_M2.append((df1['Market2'][x]-MaxM2)/MaxM2)
            DD_M3.append((df1['Market3'][x]-MaxM3)/MaxM3)
            DD_M4.append((df1['Market4'][x]-MaxM4)/MaxM4)
            x = x + 1
        portfolio.append(round(min(DD_P),3))
        m1.append(round(min(DD_M1),3))
        m2.append(round(min(DD_M2),3))
        m3.append(round(min(DD_M3),3))
        m4.append(round(min(DD_M4),3))
        df4['Measures'] = measures
        df4['Portfolio'] = portfolio
        df4['S&P 500'] = m1
        df4['Nasdaq'] = m2
        df4['ASX 200'] = m3
        df4['US Total'] = m4
        return df4
    if reason == 'risk':
        measures = ['Annual Return','Annual Volatility','Sharpe Ratio', 'Sortino Ratio']
        df4['Measures'] = measures
        portfolio.append(round(df1['P Ret'].mean()*252,3))
        m1.append(round(df1['M1 Ret'].mean()*252,3))
        m2.append(round(df1['M2 Ret'].mean()*252,3))
        m3.append(round(df1['M3 Ret'].mean()*252,3))
        m4.append(round(df1['M4 Ret'].mean()*252,3))
        df1['P Volatility'] = df1['P Ret'].rolling(window=252).std()
        df1['P Annual_Volatility'] = (df1['P Volatility'])*(252**(1/2))
        df1['M1 Volatility'] = df1['M1 Ret'].rolling(window=252).std()
        df1['M1 Annual_Volatility'] = (df1['M1 Volatility'])*(252**(1/2))
        df1['M2 Volatility'] = df1['M2 Ret'].rolling(window=252).std()
        df1['M2 Annual_Volatility'] = (df1['M2 Volatility'])*(252**(1/2))
        df1['M3 Volatility'] = df1['M3 Ret'].rolling(window=252).std()
        df1['M3 Annual_Volatility'] = (df1['M3 Volatility'])*(252**(1/2))
        df1['M4 Volatility'] = df1['M4 Ret'].rolling(window=252).std()
        df1['M4 Annual_Volatility'] = (df1['M4 Volatility'])*(252**(1/2))
        portfolio.append(round(df1['P Annual_Volatility'][len(df1['P Annual_Volatility'])-1],3))
        m1.append(round(df1['M1 Annual_Volatility'][len(df1['M1 Annual_Volatility'])-1],3))
        m2.append(round(df1['M2 Annual_Volatility'][len(df1['M2 Annual_Volatility'])-1],3))
        m3.append(round(df1['M3 Annual_Volatility'][len(df1['M3 Annual_Volatility'])-1],3))
        m4.append(round(df1['M4 Annual_Volatility'][len(df1['M4 Annual_Volatility'])-1],3))
        portfolio.append(round(((df1['P Ret'].mean()*252-0.02)/(df1['P Ret'].std()*(252**(1/2)))),3))
        m1.append(round(((df1['M1 Ret'].mean()*252-0.02)/(df1['M1 Ret'].std()*(252**(1/2)))),3))
        m2.append(round(((df1['M2 Ret'].mean()*252-0.02)/(df1['M2 Ret'].std()*(252**(1/2)))),3))
        m3.append(round(((df1['M3 Ret'].mean()*252-0.02)/(df1['M3 Ret'].std()*(252**(1/2)))),3))
        m4.append(round(((df1['M4 Ret'].mean()*252-0.02)/(df1['M4 Ret'].std()*(252**(1/2)))),3))
        dfp = df1.loc[df1['P Ret']<0]
        portfolio.append(round((df1['P Ret'].mean()*252-0.02)/(dfp['P Ret'].std()*(252**(1/2))),3))
        dfm1 = df1.loc[df1['M1 Ret']<0]
        m1.append(round((df1['M1 Ret'].mean()*252-0.02)/(dfm1['M1 Ret'].std()*(252**(1/2))),3))
        dfm2 = df1.loc[df1['M2 Ret']<0]
        m2.append(round((df1['M2 Ret'].mean()*252-0.02)/(dfm2['M2 Ret'].std()*(252**(1/2))),3))
        dfm3 = df1.loc[df1['M3 Ret']<0]
        m3.append(round((df1['M3 Ret'].mean()*252-0.02)/(dfm3['M3 Ret'].std()*(252**(1/2))),3))
        dfm4 = df1.loc[df1['M4 Ret']<0]
        m4.append(round((df1['M4 Ret'].mean()*252-0.02)/(dfm4['M4 Ret'].std()*(252**(1/2))),3))
        df4['Measures'] = measures
        df4['Portfolio'] = portfolio
        df4['S&P 500'] = m1
        df4['Nasdaq'] = m2
        df4['ASX 200'] = m3
        df4['US Total'] = m4
        return df4
    return


#this creates the app -- imports the stylesheet
app = dash.Dash(__name__,meta_tags=[{'property':'og:image','content':'https://i.ibb.co/P5RkK55/James-Charge-1.png'}])
server = app.server

#This sets the apps basic colours
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div([
    html.Div([
        html.A("Link to Stock/ETF Trading dashboard", href='https://trading-dash.herokuapp.com/', target="_blank")
        ],style={'width': '65%','display': 'inline-block'}),
    #breaking it down this way means so far there will be 2 sections to the app
    html.Div([
        html.H4('Portfolio Model Implementation and Performance'),
        dcc.Input(id='totalvalue', value=10000, type='number', debounce=True),
        html.Button('Submit', id='btn-nclicks-1', n_clicks=0),
        dcc.Graph(id='my-graph')
        ],style={'width': '60%','display': 'inline-block'}),
    html.Div([
        dcc.Upload(id='datatable-upload',
        children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '100%', 'height': '60px', 'lineHeight': '60px','borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},),
        html.H4('Holdings and their Weights'),
        dash_table.DataTable(id='datatable-upload-container')],style={'width': '20%', 'float': 'right','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Holdings and Weights Visualised'),
        dcc.Graph(id='datatable-upload-graph')
        ],style={'width': '30%', 'float': 'left','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Market Measures'),
        html.Table(id = 'my-market'),
        ],style={'width': '30%', 'float': 'middle','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Risk and Financial Measures'),
        html.Table(id = 'my-risk'),
        ],style={'width': '30%', 'float': 'right','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        dcc.Graph(id='my-MACD')
        ],style={'width': '45%', 'float': 'left','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        dcc.Graph(id='my-RSI')
        ],style={'width': '45%', 'float': 'right','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    
])


#This app callback updates the graph as per the relevant company
@app.callback(Output('my-graph','figure'),[Input('totalvalue','value'),Input('datatable-upload-container', 'data')])
def update_graph(totalvalue, rows):
    fig = Model_Display(totalvalue, 'figure', rows)
    return fig

#This app callback updates the graph as per the relevant company
@app.callback(Output('my-market','children'),[Input('totalvalue','value'),Input('datatable-upload-container', 'data')])
def update_market(totalvalue, rows):
    table = Model_Display(totalvalue, 'market', rows)
    # Header
    return html.Table([html.Tr([html.Th(col) for col in table.columns])] + [html.Tr([html.Td(table.iloc[i][col]) for col in table.columns]) for i in range(0,len(table.Portfolio))], style={'border':'solid','border-spacing':'20px'})


#This app callback updates the graph as per the relevant company
@app.callback(Output('my-risk','children'),[Input('totalvalue','value'),Input('datatable-upload-container', 'data')])
def update_risk(totalvalue, rows):
    table = Model_Display(totalvalue, 'risk', rows)
    # Header
    return html.Table([html.Tr([html.Th(col) for col in table.columns])] + [html.Tr([html.Td(table.iloc[i][col]) for col in table.columns]) for i in range(0,len(table.Portfolio))], style={'border':'solid','border-spacing':'20px'})


@app.callback(Output('datatable-upload-container', 'data'),
              Output('datatable-upload-container', 'columns'),
              Input('datatable-upload', 'contents'),
              State('datatable-upload', 'filename'))
def update_output(contents, filename):
    if contents is None:
        return [{}], []
    df = parse_contents(contents, filename)
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]


@app.callback(Output('datatable-upload-graph', 'figure'),
              Input('datatable-upload-container', 'data'))
def display_graph(rows):
    df = pd.DataFrame(rows)
    fig = go.Figure(data=[go.Pie(labels=df[df.columns[0]], values=df[df.columns[1]],insidetextorientation='radial')])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update(layout_showlegend=False)
    fig.update_layout(width=380, height = 380)
    return fig

@app.callback(Output('my-MACD', 'figure'),
              Input('datatable-upload-container', 'data'))
def display_MACD(rows):
    fig = Model_Display(10000, 'MACD', rows)
    return fig

@app.callback(Output('my-RSI', 'figure'),
              Input('datatable-upload-container', 'data'))
def display_RSI(rows):
    fig = Model_Display(10000, 'RSI', rows)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
