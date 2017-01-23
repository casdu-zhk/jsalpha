#!/usr/bin/python
# -*- coding: utf-8 -*-

# fetch.py

import tushare as ts
import pandas as pd
from WindPy import w
import os

def fetch_from_tushare(csv_dir, symbols, start_date, end_date):
    cols = ['open', 'high', 'low', 'close', 'volume', 'Adj Close']
    for s in symbols:
        df = ts.get_hist_data(s, start=start_date, end=end_date)
        df['Adj Close'] = df['close']
        df = df[cols]
        print("Saving " + s + ".csv")
        df.to_csv(csv_dir + s + ".csv")

def fetch_from_wind(csv_dir, symbols, start_date, end_date):
    w.start()
    cols = ["open", "high", "low", "close", "amt"]
    for s in symbols:
        filename = "%s%s.csv"%(csv_dir, s)
        # if os.path.exists(filename):
            # continue
        raw_data = w.wsd(s, cols, beginTime=start_date, endTime=end_date)
        dic = {}
        for data, field in zip(raw_data.Data, raw_data.Fields):
            if str.lower(str(field)) == "amt":
                field = "volume"
            dic[str.lower(str(field))] = data
        df = pd.DataFrame(dic)
        df["Adj close"] = df["close"]
        df["date"] = pd.to_datetime(raw_data.Times)
        df["date"] = df["date"].map(lambda x: x.strftime('%Y-%m-%d'))
        df = df[["date", "open", "high", "low", "close", "volume", "Adj close"]]
        assert(df.shape[0] != 0)
        df.to_csv(filename, index=False)

# if __name__ == '__main__':
    # fetch_from_wind("../csv/", ["000300.SH"], '2005-01-01', '2017-01-20')
    # fetch_from_tushare("../csv/", ["hs300"], '2015-01-01', '2015-08-31')
