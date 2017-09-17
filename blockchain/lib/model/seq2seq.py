import tensorflow as tf
from tensorflow.contrib import learn as tflearn
from tensorflow.python.ops import variable_scope
from blockchain.lib.nn import rnn
from blockchain.lib.model.base_model import BaseModel
import numpy as np
from blockchain.lib.nn.losses import losses

class Seq2Seq(BaseModel):

    def __init__(self, config):
        self.config = config
        self.model = tflearn.SKCompat(tflearn.Estimator(
            model_fn=self._model(),
            model_dir=self.config["log_dir"]
        ))
        self.bias = None
        self.weights = None


    def _model(self):
        rnn_layers = self.config["rnn_layers"]
        learning_rate = self.config["learning_rate"]
        optimizer = self.config["optimizer"]

        def _modelfn(X, y, mode):
            # Create one LSTM cell (or multiple stacked LSTM as One cell step for our seq2seq)
            cell = rnn.lstm_cells(rnn_layers)


            with variable_scope.variable_scope('linear_regression'):
                dtype = tf.float32
                output_shape = self.config["output_dim"]
                self.weights = variable_scope.get_variable(
                    'weights', [self.config["rnn_layers"][-1]["num_units"], output_shape], dtype=dtype)
                self.bias = variable_scope.get_variable('bias', [output_shape], dtype=dtype)

            # from (batch_size x len_sequence x dim_input) to (len_sequence x batch_size x dim_input)
            x_ = tf.unstack(X, axis=1, num=X.shape[1])


            if mode != tf.contrib.learn.ModeKeys.INFER:
                y_ = tf.unstack(y, axis=1, num=y.shape[1])
            else:
                y_ = None

            # if we are in learning mode, y is not None
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                # We create the decoder inputs, that are the outputs shift by 1 in the right and adding in first
                # the "GO" symbol.
                decoder_inputs = y_
                decoder_inputs[1:] = y_[:-1]
                decoder_inputs[0] = tf.zeros_like(decoder_inputs[0])
                #[tf.fill(y[-1].get_shape(), self.config["GO_SYMBOL"])]
                loop_fn = None
            else:
                batch_size = tf.shape(X)[0]
                decoder_inputs = [tf.zeros([batch_size, self.config["output_dim"]])
                                  for i in range(self.config["y_window"])]
                def loop_fn(prev, i):
                    predictions = tf.nn.xw_plus_b(prev, self.weights, self.bias)
                    return predictions


            # We run the seq2seq
            self.output, self.layers = rnn.attention_rnn_seq2seq(x_, decoder_inputs, cell, loop_function=loop_fn)

            loss = 0
            reg = 0
            output_ = []

            with variable_scope.variable_scope('label'):
                if mode != tf.contrib.learn.ModeKeys.INFER:
                    y_ = tf.unstack(y, axis=1, num=y.shape[1])

            with variable_scope.variable_scope('linear_regression'):

                if mode != tf.contrib.learn.ModeKeys.INFER:
                    for output, Y_ in zip(self.output, y_):
                        predictions = tf.nn.xw_plus_b(output, self.weights, self.bias)
                        l_ = losses.biased_mean_squared_error(Y_, predictions, self.config["bias_loss"])
                        output_.append(predictions)
                        loss += l_

                    self.output = output_

                    for tf_var in tf.trainable_variables():
                        if not ("bias" in tf_var.name.lower() or "output_" in tf_var.name.lower()):
                            reg += tf.reduce_mean(tf.nn.l2_loss(tf_var))

                        loss = loss + self.config["l2"] * reg
                else:
                    for output in self.output:
                        predictions = tf.nn.xw_plus_b(output, self.weights, self.bias)
                        # if len(predictions.get_shape()) == 2:
                        #       predictions = seq2seq.array_ops.squeeze(predictions, squeeze_dims=[1])
                        output_.append(predictions)

                    self.output = output_

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                train_op = tf.contrib.layers.optimize_loss(
                    loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
                    learning_rate=learning_rate)
            else:
                train_op = None

            return self.output, loss, train_op

        return _modelfn

    def fit(self, x, y):
        validation_monitor = tflearn.monitors.ValidationMonitor(x['val'], y['val'],
                                                              every_n_steps=self.config["print_step"],
                                                              early_stopping_rounds=self.config["early_stopping"])
        self.model.fit(x["train"], y["train"],
                      monitors=[validation_monitor],
                      batch_size=self.config["batch_size"])


    def predict(self, x):
        pred = self.model.predict(x, batch_size=self.config["batch_size"]).T
        if len(pred.shape) == 3:
            return np.concatenate(pred)
        return pred
