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
    df1 = df1.bfill(axis ='rows')
    if reason == 'figure':
        fig = go.Figure()
        king = ('Portfolio Performance of 10K Invested 2 Years Ago')
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Portfolio'], mode = 'lines', name = 'Portfolio',marker=dict(size=1, color="blue")))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Market1'], mode = 'lines', name = 'S&P 500 Benchmark',marker=dict(size=1, color="red")))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Market2'], mode = 'lines', name = 'Nasdaq Benchmark',marker=dict(size=1, color="green")))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Market3'], mode = 'lines', name = 'ASX 200 Benchmark',marker=dict(size=1, color="purple")))
        fig.add_trace(go.Scatter(x=df1.index,y=df1['Market4'], mode = 'lines', name = 'US Total Market Benchmark',marker=dict(size=1, color="orange")))
        fig.update_layout(title=king,xaxis_title="Time",yaxis_title="Portfolio Value", width=1100, height = 700)
        fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
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


def Sectors_Analysis(rows):
    df = pd.DataFrame(rows)
    counter = 0
    Industry = []
    Weight = []
    for h in df['Holdings']:
        Code = yf.Ticker(h)
        if Code.info['quoteType'] == 'EQUITY':
            Industry.append(Code.info['industry'])
            Weight.append(df['Weights'][counter])
        else:
            Industry.append('Broad Index',Code.info['market'])
            Weight.append(df['Weights'][counter])
        counter = counter + 1
    fig = go.Figure(data=[go.Pie(labels=Industry, values=Weight)])
    return fig


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
        dcc.Graph(id='my-sectors')
        ],style={'width': '30%', 'float': 'left','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Market Measures'),
        html.Table(id = 'my-market'),
        ],style={'width': '30%', 'float': 'middle','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Risk and Financial Measures'),
        html.Table(id = 'my-risk'),
        ],style={'width': '30%', 'float': 'right','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
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


@app.callback(Output('my-sectors', 'figure'),Input('datatable-upload-container', 'data'))
def display_graph(rows):
    fig = Sectors_Analysis(rows)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
