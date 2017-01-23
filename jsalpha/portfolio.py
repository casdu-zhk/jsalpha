#!/usr/bin/env python
# encoding: utf-8

from event import OrderEvent
import pandas as pd
from performance import create_sharpe_ratio, create_drawdowns

class Portfolio(object):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, 5-min, 30-min, 60 min or EOD.

    The positions DataFrame stores a time-index of the
    quantity of positions held.

    The holdings DataFrame stores the cash and total market
    holdings value of each symbol for a particular
    time-index, as well as the percentage change in
    portfolio total across bars.
    """

    def __init__(self, bars, events, start_date, initial_capital=100000.0):
        """
        Initialises the portfolio with bars and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).

        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital

        self.all_positions = self.construct_all_positions()
        self.current_positions = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def construct_all_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        d['order'] = ''
        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).

        Makes use of a MarketEvent from the events queue.
        """
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])

        # Update positions
        # ================
        dp = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        dp['datetime'] = latest_datetime

        for s in self.symbol_list:
            dp[s] = self.current_positions[s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']
        dh['order'] = self.current_holdings['order']

        for s in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[s] * \
                round(self.bars.get_latest_bar_value(s, "close"), 2)
            dh[s] = market_value
            self.current_holdings[s] = market_value
            dh['total'] += market_value
            dh['%s_position'%(s)] = self.current_positions[s]
            dh['%s_price'%(s)] = round(self.bars.get_latest_bar_value(s, 'close'), 2)

        # Append the current holdings
        self.current_holdings['total'] = dh['total']
        self.all_holdings.append(dh)

    def update_positions_from_order(self, order):
        """
        Takes a Order object and updates the position matrix to
        reflect the new position.

        Parameters:
        order - The Order object to update the positions with.
        """
        # Check whether the order is a buy or sell
        order_dir = 0
        if order.direction == 'BUY':
            order_dir = 1
        if order.direction == 'SELL':
            order_dir = -1

        # Update positions list with new quantities
        self.current_positions[order.symbol] += order_dir*order.quantity

    def update_holdings_from_order(self, order):
        """
        Takes a Order object and updates the holdings matrix to
        reflect the holdings value.

        Parameters:
        order - The Order object to update the holdings with.
        """
        # Check whether the order is a buy or sell
        self.current_holdings['order'] = order.direction
        order_dir = 0
        if order.direction == 'BUY':
            order_dir = 1
        if order.direction == 'SELL':
            order_dir = -1

        # Update holdings list with new quantities
        order_cost = round(self.bars.get_latest_bar_value(
            order.symbol, order.ask_price
        ), 2)
        cost = order_dir * order_cost * order.quantity
        self.current_holdings[order.symbol] += cost
        self.current_holdings['commission'] += order.commission
        self.current_holdings['cash'] -= (cost + order.commission)
        self.current_holdings['total'] -= (cost + order.commission)
        print order.direction, self.bars.get_latest_bar_value(order.symbol, 'close'), order.quantity

    """
        print ("%s\t%s\t%.2f\t%.2f\t%.2f")%(
            self.bars.get_latest_bar(order.symbol)[0].strftime("%Y-%m-%d"),
            order.direction,
            order_cost,
            order.quantity,
            -cost
        )
    """

    """
    def generate_safe_order(self, order):
        # Simply files an Order object as a constant quantity
        # sizing of the signal object, without risk management or
        # position sizing considerations.

        # Parameters:
        # order - The tuple containing Order information.
        safe_order = None

        symbol = order.symbol
        direction = order.direction

        mkt_quantity = order.quantity
        cur_quantity = self.current_positions[symbol]

        if direction == 'BUY' and cur_quantity == 0:
            safe_order = OrderEvent(symbol, mkt_quantity, 'BUY',
                                    ask_price=order.ask_price, safe=True)
        if direction == 'SELL' and cur_quantity == 0:
            safe_order = OrderEvent(symbol, mkt_quantity, 'SELL',
                                    ask_price=order.ask_price, safe=True)

        if direction == 'SELL' and cur_quantity > 0:
            safe_order = OrderEvent(symbol, abs(cur_quantity), 'SELL',
                                    ask_price=order.ask_price, safe=True)
        if direction == 'BUY' and cur_quantity < 0:
            safe_order = OrderEvent(symbol, abs(cur_quantity), 'BUY',
                                    ask_price=order.ask_price, safe=True)
        print direction, order.quantity
        return safe_order
    """

    def update_order(self, event):
        """
        Updates the portfolio current positions and holdings
        from a OrderEvent.
        """
        if event.type == 'ORDER':
            # event.print_order()
            cur_quantity = self.current_positions[event.symbol]
            if event.direction == 'EXIT':
                assert(cur_quantity != 0)
                assert(event.quantity == 0)
                if cur_quantity > 0:
                    event.direction = "SELL"
                else:
                    event.direction = "BUY"
                event.quantity = abs(cur_quantity)
            elif event.direction == "ALLBUY":
                current_price = self.bars.get_latest_bar_value(event.symbol, 'close')
                capital = self.current_holdings['cash']
                assert(cur_quantity == 0)
                assert(event.quantity == 0)
                event.quantity = (capital / current_price)
                event.direction = "BUY"
            elif event.direction == "ALLSELL":
                current_price = self.bars.get_latest_bar_value(event.symbol, 'close')
                capital = self.current_holdings['cash']
                assert(cur_quantity == 0)
                assert(event.quantity == 0)
                event.quantity = capital / current_price
                event.direction = "SELL"

            self.update_positions_from_order(event)
            self.update_holdings_from_order(event)

    # ========================
    # POST-BACKTEST STATISTICS
    # ========================

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0+curve['returns']).cumprod()
        for s in self.symbol_list:
            curve['%s_return'%(s)] = curve['%s_price'%(s)].pct_change()
        self.equity_curve = curve

    def output_summary_stats(self):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns, periods=252*60*6.5)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        self.equity_curve.to_csv('equity.csv')
        return stats
