import dash
import numpy as np
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
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


def Model_Display(total_value):
    url = https://github.com/jamesTrading/Portfolio-Tracker/blob/main/PortfolioModelJames.csv
    df = pd.read_csv(url)
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
    df['Units'] = amount_of_shares
    count1 = 0
    count2 = 0
    portfolio_value = []
    quick_sum = 0
    while count1 < len(df1[df['Holdings'][0]]):
        count2 = 0
        quick_sum = 0
        while count2 < len(df['Holdings']):
            quick_sum = df['Units'][count2]*df1[df['Holdings'][count2]][count1] + quick_sum
            count2 = count2 + 1
        portfolio_value.append(round(quick_sum,2))
        count1 = count1 + 1
    df1['Portfolio'] = portfolio_value
    fig = go.Figure()
    king = ('Portfolio Performance Over 2 Years')
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Portfolio'], mode = 'lines',marker=dict(size=1, color="blue"),showlegend=False))
    fig.update_layout(title=king,xaxis_title="Time",yaxis_title="Portfolio Value", width=1200, height = 700)
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
        ],style={'width': '75%','display': 'inline-block'})
])


#This app callback updates the graph as per the relevant company
@app.callback(Output('my-graph','figure'),[Input('totalvalue','value')])
def update_graph(totalvalue):
    fig = Model_Display(totalvalue)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
