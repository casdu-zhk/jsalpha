#!/usr/bin/python
# -*- coding: utf-8 -*-

# fetch.py

from __future__ import print_function
from yahoo_finance import Share
import tushare as ts
import pandas as pd
import os

def fetch_from_yahoo(csv_dir, symbols, start_date, end_date):
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    for s in symbols:
        stock = Share(s)
        data = stock.get_historical(start_date, end_date)
        df = pd.DataFrame(data)
        df = df.rename(columns={'Adj_Close': 'Adj Close'})
        df = df.drop('Symbol', 1)
        df = df.set_index('Date')
        df = df[cols]
        print("Saving " + s + ".csv")
        df.to_csv(csv_dir + s + '.csv')

def fetch_from_tushare(csv_dir, symbols, start_date, end_date):
    cols = ['open', 'high', 'low', 'close', 'volume', 'Adj Close']
    for s in symbols:
        df = ts.get_hist_data(s, start=start_date, end=end_date)
        df['Adj Close'] = df['close']
        df = df[cols]
        print("Saving " + s + ".csv")
        df.to_csv(csv_dir + s + ".csv")

if __name__ == '__main__':
    # fetch_from_yahoo("../csv/", ['AAPL'], '2015-01-01', '2015-08-31')
    fetch_from_tushare("../csv/", ["hs300"], '2015-01-01', '2015-08-31')
