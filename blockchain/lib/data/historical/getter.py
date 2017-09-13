from abc import abstractclassmethod
import pickle as pkl
import sys
import os

class Getter(object):


    def get(self, market=None, crypto_name="BCHARTS", try_cache=True):
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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            pkl.dump(df, open(dir_path+"/cache/"+market or self.market+crypto+".pkl", "wb"))
        except:
            raise Exception("Error on saving " + dir_path+"/cache/"+market or self.market+crypto+".pkl")

    def uncache(self, market, crypto):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            df=pkl.load(open(dir_path+"cache/"+market or self.market+crypto+".pkl", "rb"))
            return df
        except:
            raise Exception("Error on loading " + sys.modules[__name__].__file__+"/cache/"+market
                            or self.market+crypto+".pkl")
