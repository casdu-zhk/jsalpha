#!/usr/bin/env python
# encoding: utf-8

import pprint
import Queue as queue
import time
import numpy as np
import pandas as pd

import show

class Backtest(object):
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(
        self, csv_dir, symbol_list, initial_capital,
        heartbeat, start_date, data_handler,
        portfolio, strategy
    ):
        """
        Initialises the backtest.

        Parameters:
        csv_dir - The hard root to the CSV data directory.
        symbol_list - The list of symbol strings.
        intial_capital - The starting capital for the portfolio.
        heartbeat - Backtest "heartbeat" in seconds
        start_date - The start datetime of the strategy.
        data_handler - (Class) Handles the market data feed.
        portfolio - (Class) Keeps track of portfolio current and prior positions.
        strategy - (Class) Generates signals based on market data.
        """
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.data_handler_cls = data_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy

        self.events = queue.Queue()

        self.orders = 0
        self.num_strats = 1

        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        self.data_handler = self.data_handler_cls(self.events, self.csv_dir, self.symbol_list)
        self.strategy = self.strategy_cls(self.data_handler, self.events)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, self.start_date,
                                            self.initial_capital)

    def _run_backtest(self):
        """
        Executes the backtest.
        """
        while True:
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
            else:
                break

            # Handle the events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)

                        elif event.type == 'ORDER':
                            self.portfolio.update_order(event)
                            self.orders += 1

            time.sleep(self.heartbeat)

    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()

        print("Creating summary stats...")
        stats = self.portfolio.output_summary_stats()

        print("Creating equity curve...")
        print(self.portfolio.equity_curve.tail(10))
        pprint.pprint(stats)

        print("Orders: %s" % self.orders)

    def _show_result(self, period="daily"):
        df = pd.read_csv("./equity.csv")
        df.index = pd.to_datetime(df["datetime"], format="%Y-%m-%d")
        df.drop(df.index[0], inplace=True)
        if len(self.symbol_list) == 1:
            symbol = self.symbol_list[0]
            benchmark = df["%s_price"%(symbol)]
            benchmark_daily_return = benchmark.pct_change()
            benchmark_daily_return[0] = 0.0
            daily_return = df["returns"]
            show.show_result(daily_return, period, benchmark_daily_return, symbol)
        else:
            raise NotImplementedError

    def simulate_trading(self, plot=False, period="daily"):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        print("Date\t\tOrder\tPrice\tQuantity\tCost")
        self._run_backtest()
        self._output_performance()
        if plot:
            self._show_result(period)
