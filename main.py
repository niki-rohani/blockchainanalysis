from blockchain.lib.data.historical import *
import ipdb

api_key = "f9698sEkpQ_kUyz7ThBs"

qg = QuandlGetter(api_key)

df = qg.get(market=MARKET_KRAKEN)

df.head()
