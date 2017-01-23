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
from fetch import fetch_from_wind

class Average_Momentum(Strategy):

    def __init__(self, bars, events, short_window=10, long_window=60):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
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
                bars = self.bars.get_latest_bars_values(symbol, "close", self.long_window)
                bars = pd.Series(bars)
                if bars.shape[0] < self.long_window:
                    continue
                past_return_long = bars.pct_change().mean()
                bars = self.bars.get_latest_bars_values(symbol, "close", self.short_window)
                bars = pd.Series(bars)
                past_return_short = bars.pct_change().mean()

                bars = self.bars.get_latest_bars_values(symbol, "volume", self.long_window)
                bars = pd.Series(bars)
                if bars.shape[0] < self.long_window:
                    continue
                past_volume_long = bars.pct_change().mean()
                bars = self.bars.get_latest_bars_values(symbol, "volume", self.short_window)
                bars = pd.Series(bars)
                past_volume_short = bars.pct_change().mean()

                if self.bought[symbol] == "OUT" and past_return_short > past_return_long and past_volume_short > past_volume_long:
                    order = OrderEvent(symbol, "ALLBUY")
                    self.events.put(order)
                    self.bought[symbol] = "LONG"
                if self.bought[symbol] == "LONG" and past_return_long > past_return_short:
                    order = OrderEvent(symbol, "EXIT")
                    self.events.put(order)
                    self.bought[symbol] = "OUT"

                """
                if self.bought[symbol] == "OUT": # 空仓
                    if past_return_long < past_return_short:
                        order = OrderEvent(symbol, "ALLBUY")
                        self.events.put(order)
                        self.bought[symbol] = "LONG"
                    else:
                        order = OrderEvent(symbol, "ALLSELL")
                        self.events.put(order)
                        self.bought[symbol] = "SHORT"

                elif self.bought[symbol] == "LONG" and past_return_long > past_return_short: # 多头翻转
                    # 先平仓
                    order = OrderEvent(symbol, "EXIT")
                    self.events.put(order)
                    self.bought[symbol] = "OUT"

                    # 再开仓做空
                    order = OrderEvent(symbol, "ALLSELL")
                    self.events.put(order)
                    self.bought[symbol] = "SHORT"

                elif self.bought[symbol] == "SHORT" and past_return_long < past_return_short: # 空头翻转
                    # 先平仓
                    order = OrderEvent(symbol, "EXIT")
                    self.events.put(order)
                    self.bought[symbol] = "OUT"

                    # 再开仓做多
                    order = OrderEvent(symbol, "ALLBUY")
                    self.events.put(order)
                    self.bought[symbol] = "LONG"
                """


if __name__ == "__main__":
    csv_dir = '../csv/'
    start_date = '2005-01-01'
    end_date = '2017-01-20'
    symbol_list = ['000300.SH']
    fetch_from_wind(csv_dir, symbol_list, start_date, end_date)
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
