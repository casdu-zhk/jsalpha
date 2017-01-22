#!/usr/bin/env python
# encoding: utf-8

import datetime
import numpy as np
import pandas as pd
import sys

sys.path.append("../jsalpha/")

from backtest import Backtest
from data import HistoricCSVDataHandler
from event import OrderEvent
from portfolio import Portfolio
from strategy import Strategy
from fetch import fetch_from_tushare

class Average_Momentum(Strategy):

    def __init__(self, bars, events, look_back=12):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.look_back = look_back
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_signals(self, event):
        if event.type == "MARKET":
            for symbol in self.symbol_list:
                dt = self.bars.get_latest_bar_datetime(symbol)
                bars = self.bars.get_latest_bars_values(symbol, "close", self.look_back)
                bars = pd.Series(bars)
                if bars.shape[0] < self.look_back:
                    continue
                past_return = bars.pct_change().mean()

                if self.bought[symbol] == "OUT" and past_return > 0: # 空仓
                    order = OrderEvent(symbol, "ALLBUY")
                    self.events.put(order)
                    self.bought[symbol] = "LONG"
                if self.bought[symbol] == "LONG" and past_return < 0:
                    order = OrderEvent(symbol, "EXIT")
                    self.events.put(order)
                    self.bought[symbol] = "OUT"

if __name__ == "__main__":
    csv_dir = '../csv/'
    start_date = '2015-01-01'
    end_date = '2017-01-20'
    symbol_list = ['hs300']
    # fetch_from_tushare(csv_dir, symbol_list, start_date, end_date)
    initial_capital = 100000.0
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    heartbeat = 0.0

    backtest = Backtest(csv_dir,
                        symbol_list,
                        initial_capital,
                        heartbeat,
                        start_date,
                        HistoricCSVDataHandler,
                        Portfolio,
                        Average_Momentum)

    backtest.simulate_trading(plot=True)
