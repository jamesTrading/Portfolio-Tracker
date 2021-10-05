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
    if reason == 'returns':
        outputlist = []
        Month1 = round(((df1['Portfolio'][len(df1['Portfolio'])-1] - df1['Portfolio'][len(df1['Portfolio'])-22])/df1['Portfolio'][len(df1['Portfolio'])-22])*100,3)
        Month3 = round(((df1['Portfolio'][len(df1['Portfolio'])-1] - df1['Portfolio'][len(df1['Portfolio'])-63])/df1['Portfolio'][len(df1['Portfolio'])-63])*100,3)
        Month12 = round(((df1['Portfolio'][len(df1['Portfolio'])-1] - df1['Portfolio'][len(df1['Portfolio'])-252])/df1['Portfolio'][len(df1['Portfolio'])-252])*100,3)
        outputlist.append(("1 Month Portfolio Return: ",Month1, "%"))
        outputlist.append(("3 Month Portfolio Return: ",Month3, "%"))
        outputlist.append(("1 Year Portfolio Return: ",Month12, "%"))
        return outputlist        
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
    if reason == 'market':
        df4 = pd.DataFrame()
        measures = ['Best Month','Worst Month','Avg. Gain in Up','Avg. Loss in Down','Positive Months','Downside Deviation','Worst 1 Day']
        portfolio = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        market = [1,2,3,4,5,6,7]
        df4['Measures'] = measures
        df4['Portfolio'] = portfolio
        df4['Marktet'] = market
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
        dash_table.DataTable(id='datatable-upload-container'),
        html.H4('Returns of the Portfolio'),
        html.Table(id = 'my-returns')],style={'width': '20%', 'float': 'right','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Holdings and Weights Visualised'),
        dcc.Graph(id='datatable-upload-graph')
        ],style={'width': '30%', 'float': 'left','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Market Measures'),
        dash_table.DataTable(id='datatable-market-container'),
        ],style={'width': '30%', 'float': 'middle','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
])


#This app callback updates the graph as per the relevant company
@app.callback(Output('my-graph','figure'),[Input('totalvalue','value'),Input('datatable-upload-container', 'data')])
def update_graph(totalvalue, rows):
    fig = Model_Display(totalvalue, 'figure', rows)
    return fig


#This app callback updates the graph as per the relevant company
@app.callback(Output('my-returns','children'),[Input('totalvalue','value'),Input('datatable-upload-container', 'data')])
def update_returns(totalvalue, rows):
    outputlist = Model_Display(totalvalue, 'returns', rows)
    # Header
    return [html.Tr(html.Td(output)) for output in outputlist]

#This app callback updates the graph as per the relevant company
@app.callback(Output('datatable-market-container','columns'),Output('datatable-market-container', 'data'),[Input('totalvalue','value'),Input('datatable-upload-container', 'data')])
def update_market(totalvalue, rows):
    df = Model_Display(totalvalue, 'market', rows)
    print(df)
    # Header
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]

@app.callback(Output('datatable-upload-container', 'data'),
              Output('datatable-upload-container', 'columns'),
              Input('datatable-upload', 'contents'),
              State('datatable-upload', 'filename'))
def update_output(contents, filename):
    if contents is None:
        return [{}], []
    df = parse_contents(contents, filename)
    print(df)
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]


@app.callback(Output('datatable-upload-graph', 'figure'),
              Input('datatable-upload-container', 'data'))
def display_graph(rows):
    df = pd.DataFrame(rows)

    if (df.empty or len(df.columns) < 1):
        return {
            'data': [{
                'x': [],
                'y': [],
                'type': 'bar'
            }]
        }
    return {
        'data': [{
            'x': df[df.columns[0]],
            'y': df[df.columns[1]],
            'type': 'bar'
        }]
    }


if __name__ == '__main__':
    app.run_server(debug=True)
