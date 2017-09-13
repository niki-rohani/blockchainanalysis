import pandas as pd
import numpy as np

class Dataset(object):

    @staticmethod
    def create_series_dataset(x, y, x_column, label, lookback_len=10, ratio=[0.2, 0.2], shuffle=False, y_window=1):
        """
        Create a dataset from x and y
        :param x:
        :param y:
        :return:
        """
        x = x[x_column]
        y = y[[label]]
        y["y"] = y[label]
        y=y[["y"]]
        df = pd.concat([x, y], axis=1, join="inner")

        N = df["y"].shape[0]

        if y_window > 1:
            x = np.array([np.array(df[x_column])[b:b + lookback_len].astype(np.float32)
                          for b in range(0, N - lookback_len - y_window)])
            y = np.array([np.array(df[["y"]])[b:b+y_window].astype(np.float32)
                          for b in range(lookback_len, N - y_window)])
        else:
            x = np.array([np.array(df[x_column])[b:b + lookback_len].astype(np.float32)
                          for b in range(0, N - lookback_len)])
            y = np.array([np.array(df["y"]).astype(np.float32)[b]
                          for b in range(lookback_len, N)])

        def split_data(data, val_size=0.1, test_size=0.1):
            """
            splits data to training, validation and testing parts
            """
            ntest = int(round(len(data) * (1 - test_size)))
            nval = int(round(len(data[:ntest]) * (1 - val_size)))

            idx_split = [i for i in range(len(data))]
            if shuffle:
                np.random.shuffle(idx_split)
            train, val, test = idx_split[:nval], idx_split[nval:ntest], idx_split[ntest:]

            return train, val, test

        idx = split_data(x, ratio[0], ratio[1])

        xtrain, xval, xtest = x[idx[0]], x[idx[1]], x[idx[2]]
        ytrain, yval, ytest = y[idx[0]], y[idx[1]], y[idx[2]]

        return dict(train=xtrain, val=xval, test=xtest), dict(train=ytrain, val=yval, test=ytest)

