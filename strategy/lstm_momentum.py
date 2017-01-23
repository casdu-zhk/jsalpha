from datetime import datetime
import numpy as np
import pandas as pd
import sys
import os
import argparse
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import theano
import matplotlib.pyplot as plt

sys.path.append("../jsalpha/")

from backtest import Backtest
from data import HistoricCSVDataHandler
from event import OrderEvent
from portfolio import Portfolio
from strategy import Strategy
from fetch import fetch_from_wind

class LSTM_momentum(Strategy):

    def __init__(self, bars, events, look_back=7, look_ahead=5,
                buy_threshold=0.0, train_date="2010-01-01", test_date="2012-01-01", train_mode=False):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.position = {}
        self.buy_threshold = buy_threshold
        self.exit_day = None
        self.pred_queue = []
        self.pred = {}
        self.orders = 0
        self.correct = 0
        self.enter_day = None
        self.buy_price = None
        self.pred_profit = None
        self.train_mode = train_mode
        self.temp_file = "./temp.txt"
        for symbol in self.symbol_list:
            self.position[symbol] = 'OUT'

        self.model_path = "./lstm_model.h5"
        self.scaler_path = "./scaler.pkl"
        if self.train_mode:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.model = None
            self.train_date = datetime.strptime(train_date, "%Y-%m-%d")
            self.test_date = datetime.strptime(test_date, "%Y-%m-%d")
            self.train_flag = False
            self.test_flag = False
        else:
            print("Loading model...")
            theano.config.compute_test_value = 'off'
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

    def create_sequence_dataset(self, dataset, look_back=12):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def train_LSTM_model(self, train_data):
        print("======================training model=======================")
        assert(not np.isnan(train_data).any())
        np.random.seed(1234)
        train_data= train_data.reshape(-1, 1)
        train_data = self.scaler.fit_transform(train_data)
        print "Training sample data size: ", train_data.shape
        trainX, trainY = self.create_sequence_dataset(train_data, self.look_back)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        joblib.dump(self.scaler, self.scaler_path)
        print("Training LSTM model...")
        # create and fit the LSTM network
        self.model = Sequential()
        self.model.add(LSTM(4, input_dim=self.look_back))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(trainX, trainY, nb_epoch=20, batch_size=2, verbose=2)
        trainPredict = self.model.predict(trainX)
        train_return = pd.Series(trainY).pct_change(self.look_ahead)
        train_return_pred = pd.Series(np.squeeze(trainPredict)).pct_change(self.look_ahead)
        print("Training accurary: %.2f%%"%(accuracy_score(train_return > 0, train_return_pred > 0) * 100))
        self.model.save(model_path)

    def test_model(self, test_data):
        assert(not np.isnan(test_data).any())
        test_data = test_data.reshape(-1, 1)
        print("======================testing model========================")
        test_data = self.scaler.transform(test_data)
        print "Testing sample data size: ", test_data.shape
        testX, testY = self.create_sequence_dataset(test_data, self.look_back)
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        testPredict = self.model.predict(testX)
        test_return = pd.Series(testY).pct_change(self.look_ahead)
        test_return_pred = pd.Series(np.squeeze(testPredict)).pct_change(self.look_ahead)
        print("Testing accurary: %.2f%%"%(accuracy_score(test_return > 0, test_return_pred > 0) * 100))

    def calculate_signals(self, event):
        if event.type == "MARKET":
            for symbol in self.symbol_list:
                # today
                today = self.bars.get_latest_bar_datetime(symbol)
                today = today.to_pydatetime()

                if self.train_mode:
                    if today < self.train_date:
                        continue
                    if today > self.train_date and not self.train_flag:
                        history_price = self.bars.get_latest_bars_values(symbol, 'close', N=(today-backtest.start_date).days)
                        self.train_LSTM_model(history_price)
                        self.train_flag = True

                    if today < self.test_date:
                        continue
                    if today > self.test_date and not self.test_flag:
                        history_price = self.bars.get_latest_bars_values(symbol, 'close', N=(today-self.train_date).days)
                        self.test_model(history_price)
                        self.test_flag = True


                past_prices = self.bars.get_latest_bars_values(symbol, 'close', N=self.look_back)
                if past_prices.shape[0] < self.look_back:
                    continue
                past_prices = past_prices.reshape(-1, 1)
                past_prices = self.scaler.transform(past_prices)
                past_prices = np.reshape(past_prices, (1, 1, past_prices.shape[0]))

                today_price = self.bars.get_latest_bar_value(symbol, 'close')

                predict = self.model.predict(past_prices)[0][0] # predicted price after look_ahead days
                proba = self.model.predict_proba(past_prices)[0][0] # probability of prediction

                # predict is the predicted value after $look_ahead$ trading days
                # today is the datetime of this trading day
                # pred_value is the predicted value before $look_ahead$ trading days
                # pred_day is trading date when today's predicted value is made
                self.pred_queue.append((predict, today))
                print past_prices, today
                if len(self.pred_queue) == self.look_ahead:
                    (pred_value, pred_day) = self.pred_queue.pop(0)
                    self.pred[today] = pred_value
                else:
                    # less than $look_ahead + 1$ trading days, do not trade
                    continue

                print self.scaler.transform(np.array([today_price]).reshape(-1, 1))[0], pred_value


                if self.pred.has_key(pred_day):
                    self.orders += 1
                    if (pred_value - self.pred[pred_day] > 0) and (today_price > self.bars.get_latest_bars_values(symbol, 'close', N=self.look_ahead)[0]):
                        self.correct += 1
                    if (pred_value - self.pred[pred_day] < 0) and (today_price < self.bars.get_latest_bars_values(symbol, 'close', N=self.look_ahead)[0]):
                        self.correct += 1
                    line = "%d\t%d\t%.2f%%"%(self.correct, self.orders, self.correct * 1.0 / self.orders * 100)
                    print(line)
                    with open(self.temp_file, 'a') as f:
                        f.write(line + '\n')

                    line = "%s,%.2f,%.6f"%(today.strftime("%Y-%m-%d"), today_price, pred_value)
                    with open("./res.csv", 'a') as f:
                        f.write(line + "\n")


                if (self.position[symbol] == "LONG"):
                    # exit if $look_ahead$ tradings days come to the end no matter win or loss or profit is greater than predicted
                    if (pred_day == self.enter_day):
                        line = "EXIT\t%s\t%.2f\t%.2f"%(today.strftime("%Y-%m-%d"), today_price, today_price - self.buy_price)
                        print(line)
                        # with open(self.temp_file, 'a') as f:
                            # f.write(line + '\n')
                        order = OrderEvent(symbol, "EXIT")
                        self.events.put(order)
                        self.position[symbol] = "OUT"

                if proba > self.buy_threshold and self.pred.has_key(today):
                    # if position is empty and predicted value is greater than today' predicted value, then buy
                    if self.position[symbol] == "OUT" and predict - pred_value > 0:
                        self.enter_day = today
                        self.buy_price = today_price
                        self.pred_profit = predict / pred_value - 1
                        line = "BUY\t%s\t%.f\t%.2f"%(today.strftime("%Y-%m-%d"), today_price, proba * 100)
                        print(line)
                        # with open(self.temp_file, 'a') as f:
                            # f.write(line + '\n')
                        order = OrderEvent(symbol, "ALLBUY")
                        self.events.put(order)
                        self.position[symbol] = "LONG"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", help="securities", default="000300.SH", type=str)
    parser.add_argument("--s", help="start date", default="2002-01-01", type=str)
    parser.add_argument("--e", help="end date", default="2017-01-20", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    csv_dir = "../csv/"
    args = get_args()
    symbol_list = args.symbols.split(',')
    fetch_from_wind(csv_dir, symbol_list, args.s, args.e)
    start_date = datetime.strptime(args.s, "%Y-%m-%d")
    initial_capital = 100000.0
    heartbeat = 1.0

    backtest = Backtest(csv_dir,
                        symbol_list,
                        initial_capital,
                        heartbeat,
                        start_date,
                        HistoricCSVDataHandler,
                        Portfolio,
                        LSTM_momentum)

    backtest.simulate_trading(plot=True)
