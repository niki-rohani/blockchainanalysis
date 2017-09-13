import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.ops import losses_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from blockchain.lib.nn import rnn
from blockchain.lib.model import BaseModel

class Seq2Seq(object):

    def __init__(self, config):
        self.config = config
        self.model = tflearn.SKCompat(tflearn.Estimator(
            model_fn=self._model(),
            model_dir=self.config["log_dir"]
        ))


    def _model(self):
        rnn_layers = self.config["rnn_layers"]
        learning_rate = self.config["learning_rate"]
        optimizer = self.config["optimizer"]

        def _modelfn(X, y):
            # Create one LSTM cell (or multiple stacked LSTM as One cell step for our seq2seq)
            cell = rnn.lstm_cells(rnn_layers)

            # from (batch_size x len_sequence x dim_input) to (len_sequence x batch_size x dim_input)
            x_ = tf.unstack(X, axis=1, num=X.shape[1])

            # if we are in learning mode, y is not None
            if y is not None:
                # We create the decoder inputs, that are the outputs shift by 1 in the right and adding in first
                # the "GO" symbol.
                decoder_inputs = [tf.fill(y[-1].get_shapes(), self.config["GO_SYMBOL"])]
                for input in decoder_inputs:
                    decoder_inputs.append(input)
                y_ = tf.unstack(y, axis=1, num=y.shape[1])
                decoder_inputs = tf.unstack(decoder_inputs, axis=1, num=decoder_inputs.shape[1])
                loop_fn = None
            else:
                y_ = None
                decoder_inputs = [tf.fill([None, self.config["y_window"],
                                           self.config["output_dim"]], self.config["GO_SYMBOL"])]

            # We run the seq2seq
            self.output, self.layers = rnn.attention_rnn_seq2seq(x_, decoder_inputs, cell)

            loss = 0
            i = 0
            output_ = []

            with variable_scope.variable_scope('linear_regression'):
                scope_name = variable_scope.get_variable_scope().name
                dtype = self.output[0].dtype.base_dtype
                output_shape = 1
                weights = variable_scope.get_variable(
                    'weights', [self.output[0].get_shape()[1], output_shape], dtype=dtype)
                bias = variable_scope.get_variable('bias', [output_shape], dtype=dtype)

                if y is not None:
                    # Set up the requested initialization.
                    summary.histogram('%s.weights' % scope_name, weights)
                    summary.histogram('%s.bias' % scope_name, bias)

                    for output, Y_ in zip(self.output, y_):
                        summary.histogram('%s.x_%s' % (scope_name, str(i)), output)
                        summary.histogram('%s.y_%s' % (scope_name, str(i)), Y_)
                        prediction, l_ = losses_ops.mean_squared_error_regressor(output, Y_, weights, bias)
                        output_.append(prediction)
                        i+=1
                        loss += l_
                    self.output = output_
                    loss = loss / (i+1)
                else:
                    for output in self.output:
                        predictions = tf.nn.xw_plus_b(output, weights, bias)
                        if len(predictions.get_shape()) == 2:
                            predictions = seq2seq.array_ops.squeeze(predictions, squeeze_dims=[1])
                        output_.append(predictions)

                    self.output = output_

            train_op = tf.contrib.layers.optimize_loss(
                loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
                learning_rate=learning_rate)
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
        return self.model.predict(x, batch_size=self.config["batch_size"])
