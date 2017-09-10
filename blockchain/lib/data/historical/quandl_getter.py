from . import Getter
import quandl

MARKET_KRAKEN="KRAKEN"

class QuandlGetter(Getter):


    def __init__ (self, api_key):
        quandl.ApiConfig.api_key = api_key

    def _get(self, market, crypto_name=None):
        quandl_id = "BCHARTS/"+market+"USD"
        df = quandl.get(quandl_id, returns="pandas")
        return df