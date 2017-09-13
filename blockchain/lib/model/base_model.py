from abc import abstractclassmethod

class BaseModel(object):

    @abstractclassmethod
    def fit(self, x, y):
        pass

    @abstractclassmethod
    def predict(self, x):
        pass

    @abstractclassmethod
    def _model(self):
        pass