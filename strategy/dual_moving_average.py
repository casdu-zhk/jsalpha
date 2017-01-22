import datetime
import numpy as np
import sys

sys.path.append("../jsalpha/")

from backtest import Backtest
from data import HistoricCSVDataHandler
from event import OrderEvent
from portfolio import Portfolio
from strategy import Strategy
from fetch import fetch_from_yahoo, fetch_from_tushare

class MovingAverageCrossStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 100/400 periods respectively.
    """

    def __init__(self, bars, events, short_window=10, long_window=30):
        """
        Initialises the buy and hold strategy.

        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.short_window = short_window
        self.long_window = long_window

        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_signals(self, event):
        """
        Generates a new set of signals based on the MAC
        SMA with the short window crossing the long window
        meaning a long entry and vice versa for a short entry.

        Parameters
        event - A MarketEvent object.
        """
        if event.type == 'MARKET':
            for symbol in self.symbol_list:
                dt = self.bars.get_latest_bar_datetime(symbol)
                # print("%s\t%.2f"%(dt.strftime("%Y-%m-%d"), self.bars.get_latest_bar_value(symbol, 'close')))
                bars = self.bars.get_latest_bars_values(symbol, "close", N=self.long_window)
                if bars.shape[0] < self.long_window:
                    continue

                if bars is not None and bars != []:
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])
                    # print("%s\t%.2f\t%.2f\t%.2f\t%.2f"%(dt.strftime("%Y-%m-%d"),
                    #                                   self.bars.get_latest_bar_value(symbol, 'close'), short_sma, long_sma, bars[0]))
                    if short_sma > long_sma and self.bought[symbol] == "OUT":
                        print("%s\t%.2f\t%.2f\tbuy"%(dt.strftime("%Y-%m-%d"), short_sma, long_sma))
                        order = OrderEvent(symbol, "ALLBUY")
                        self.events.put(order)
                        self.bought[symbol] = "LONG"

                    elif short_sma < long_sma and self.bought[symbol] == "LONG":
                        print("%s\t%.2f\t%.2f\tsell"%(dt.strftime("%Y-%m-%d"), short_sma, long_sma))
                        order = OrderEvent(symbol, "EXIT")
                        self.events.put(order)
                        self.bought[symbol] = "OUT"

if __name__ == "__main__":
    csv_dir = '../csv/'
    start_date = '2014-01-22'
    end_date = '2017-01-20'
    symbol_list = ['000402']
    # fetch_from_yahoo(csv_dir, symbol_list, start_date, end_date)
    fetch_from_tushare(csv_dir, symbol_list, start_date, end_date)
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
                        MovingAverageCrossStrategy)

    backtest.simulate_trading(plot=True)