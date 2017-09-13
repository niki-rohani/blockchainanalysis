from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from blockchain.lib.data.dataset import Dataset
from blockchain.lib.nn import rnn
from blockchain.lib.model import BaseModel

class LSTMModel(BaseModel):

    def __init__(self, config):
        self.config = config
        self.model = tflearn.SKCompat(tflearn.Estimator(
            model_fn=self._model(),
            model_dir=self.config["log_dir"]
        ))


    def _model(self):
        num_units = self.config["num_unit"]
        rnn_layers = self.config["rnn_layers"]
        dense_layers = self.config["dense_layers"]
        learning_rate = self.config["learning_rate"]
        optimizer = self.config["optimizer"]

        """
        Creates a deep model based on:
            * stacked lstm cells
            * an optional dense layers
        :param num_units: the size of the cells.
        :param rnn_layers: list of int or dict
                             * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                             * list of dict: [{steps: int, keep_prob: int}, ...]
        :param dense_layers: list of nodes for each layer
        :return: the model definition
        """

        def dnn_layers(input_layers, layers):
            if layers and isinstance(layers, dict):
                return tflayers.stack(input_layers, tflayers.fully_connected,
                                      layers['layers'],
                                      activation=layers.get('activation'),
                                      dropout=layers.get('dropout'))
            elif layers:
                return tflayers.stack(input_layers, tflayers.fully_connected, layers)
            else:
                return input_layers

        def _lstm_model(X, y):
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(rnn.lstm_cells(rnn_layers), state_is_tuple=True)
            x_ = tf.unstack(X, axis=1, num=num_units)
            output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
            output = dnn_layers(output[-1], dense_layers)
            prediction, loss = tflearn.models.linear_regression(output, y)
            train_op = tf.contrib.layers.optimize_loss(
                loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
                learning_rate=learning_rate)
            return prediction, loss, train_op

        return _lstm_model

    def fit(self, x, y):
        x, y = Dataset.create_series_dataset(x, y, self.config["column"],
                                             self.config["label"],
                                             self.config["len_sequence"],
                                             ratio=self.config["ratio_train_val_test"])
        validation_monitor = tflearn.monitors.ValidationMonitor(x['val'], y['val'],
                                                              every_n_steps=self.config["print_step"],
                                                              early_stopping_rounds=self.config["early_stopping"])
        self.model.fit(x["train"], y["train"],
                      monitors=[validation_monitor],
                      batch_size=self.config["batch_size"])

    def predict(self, x):
        pass