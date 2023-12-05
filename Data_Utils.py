import yfinance as yf
import pandas as pd
from scipy.signal import argrelextrema
from ta import add_all_ta_features
import numpy as np

class StockDataUtils:
    def __init__(self, tickers, start_date, end_date):
        """
                Initialize StockDataUtils object.

                Args:
                - tickers (list): List of stock ticker symbols.
                - start_date (str): Start date for fetching stock data in 'YYYY-MM-DD' format.
                - end_date (str): End date for fetching stock data in 'YYYY-MM-DD' format.
                """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.fetch_stock_data()
        self.create_features()
        self.label_local_extremes()
        self.data.index = range(len(self.data))
        self.buy_points_index = self.data[self.data['Extremes']==1].index.to_numpy()

    def fetch_stock_data(self):
        """
                Fetch stock data for the provided tickers within the specified date range.

                Returns:
                - pandas.DataFrame: Combined stock data for the given tickers.
        """
        combined_data = pd.DataFrame()
        for ticker in self.tickers:
            data = yf.download(ticker, start=self.start_date, end=self.end_date)
            data['Ticker'] = ticker
            combined_data = pd.concat([combined_data, data])
        return combined_data

    def label_local_extremes(self):
        """
                Label local extremes in stock data.
        """
        self.data['Extremes'] = 0
        for ticker in self.tickers:
            ticker_data = self.data[self.data['Ticker'] == ticker]
            close_prices = ticker_data['Close'].values
            local_min_indices = argrelextrema(close_prices, comparator=np.less_equal, order=3)[0]

            self.data.loc[(self.data['Ticker'] == ticker) & (self.data.index.isin(ticker_data.index[local_min_indices])), 'Extremes'] = 1

    def create_features(self):
        """
                Generate additional features from stock data.
        """
        self.data = add_all_ta_features(self.data, open="Open", high="High", low="Low", close="Close", volume="Volume")


        # Moving averages
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['MA_15'] = self.data['Close'].rolling(window=15).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()

        # Volatility measures
        self.data['STD_20'] = self.data['Close'].rolling(window=20).std()

        # Momentum indicators
        self.data['RSI'] = self.data['momentum_rsi']
        self.data['Stochastic_Oscillator'] = self.data['momentum_stoch']

        # clean na values
        self.data = self.data.fillna(0)
        self.data = self.data.replace([np.inf, -np.inf], 0)

        self.data = self.data.drop(['Open','Low','High'],axis=1)


def create_data_for_algo(ticker_list,start_date,end_date):
    """
        Create training data for an algorithm based on stock data.

        Args:
        - ticker_list (list): List of stock ticker symbols.
        - start_date (str): Start date for fetching stock data in 'YYYY-MM-DD' format.
        - end_date (str): End date for fetching stock data in 'YYYY-MM-DD' format.

        Returns:
        - numpy.ndarray: Training data for the algorithm (features).
        - numpy.ndarray: Labels/targets for the algorithm.
        """

    stock_utils = StockDataUtils([ticker_list[0]], start_date, end_date)
    stock_utils.data.drop('Ticker', axis=1, inplace=True)
    training_data = np.array(stock_utils.data)

    for index,ticker in enumerate(ticker_list):
        if index == 0:
            pass
        else:
            # Create StockDataUtils instance
            stock_utils = StockDataUtils([ticker],start_date,end_date)
            stock_utils.data.drop('Ticker', axis=1, inplace=True)
            training_data = np.concatenate([training_data,np.array(stock_utils.data)],axis=0)

    return training_data[:,:training_data.shape[1]-1],training_data[:,-1]
