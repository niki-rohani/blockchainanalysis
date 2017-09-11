from . import Getter
import quandl
import numpy as np
import pandas as pd
import datetime

CRYPTO_BTC_ETH = "BTC_ETH"
CRYPTO_USD_ETH = "USD_ETH"

class PoliniexGetter(Getter):


    def __init__ (self, period=86400, start_date='2014-01-01'):
        self.base_polo_url = \
            'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
        self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')  # get data from the start of 2014
        self.end_date = datetime.datetime.now()  # up until today
        self.period = period  # pull daily data (86,400 seconds per day)
        self.market="poliniex"

    def _get(self, market=None, crypto_name=None):
        crypto_id = crypto_name
        df = self._get_crypto_data(crypto_id)
        df = df.replace(0, np.nan)
        df["price"] = df["weightedAverage"]
        df["market"] = self.market
        df["crypto_name"] = crypto_name.split("_")[1]
        df["base"] = crypto_name.split("_")[0].replace("USDT", "USD")
        return df

    def _get_crypto_data(self, poloniex_pair):
        '''Retrieve cryptocurrency data from poloniex'''
        json_url = self.base_polo_url.format(poloniex_pair,
                                             self.start_date.timestamp(),
                                             self.end_date.timestamp(),
                                             self.period)

        data_df = pd.read_json(json_url, poloniex_pair)
        data_df = data_df.set_index('date')
        return data_df