from . import Getter
import quandl
import numpy as np

MARKET_KRAKEN="KRAKEN"
MARKET_COINBASE='COINBASE'
MARKET_BITSTAMP='BITSTAMP'
MARKET_ITBIT='ITBIT'

class QuandlGetter(Getter):


    def __init__ (self, api_key):
        quandl.ApiConfig.api_key = api_key

    def _get(self, market, crypto_name=None):
        quandl_id = "BCHARTS/"+market+"USD"
        df = quandl.get(quandl_id, returns="pandas")
        df = df.replace(0, np.nan)
        df["price"] = df["Weighted Price"]
        df["market"] = market
        df["crypto_name"] = "BTC"
        df["base"] = "USD"
        return df