from abc import abstractclassmethod
import pickle as pkl

class Getter(object):


    def get(self, market, crypto_name="BCHARTS", try_cache=True):
        """
        Abstract method for getting a crypto name historical data on a market
        :param crypto_name:
        :param market:
        :return: pandas
        """
        if try_cache:
            try:
                df=self.uncache(market, crypto_name)
                return df
            except:
                pass

        df = self._get(market, crypto_name)
        self.cache(df, market, crypto_name)
        return df

    @abstractclassmethod
    def _get(self, market, crypto_name):
        pass

    def cache(self, df, market, crypto):
        try:
            pkl.dump(df, open("cache/"+market+crypto+".pkl", "wb"))
        except:
            raise ("Error on saving " + market+crypto)

    def uncache(self, market, crypto):
        try:
            df=pkl.load(open("cache/"+market+crypto+".pkl"), "rb")
            return df
        except:
            raise ("Error on loading " + market+crypto)
