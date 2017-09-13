import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.ops import losses_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary

def lstm_cells(layers):

    return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(
            layer['num_units'], state_is_tuple=False
        ),
        layer['keep_prob']
    ) if layer.get('keep_prob') else tf.contrib.rnn.BasicLSTMCell(
        layer['num_units'],
        state_is_tuple=False
    ) for layer in layers
                                        ], state_is_tuple=False)


def attention_rnn_seq2seq(encoder_inputs,
                          decoder_inputs,
                          cell,
                          dtype=dtypes.float32,
                          scope=None):
    """Basic RNN sequence-to-sequence model.

    This model first runs an RNN to encode encoder_inputs into a state vector,
    then runs decoder, initialized with the last encoder state, on decoder_inputs.
    Encoder and decoder use the same RNN cell type, but don't share parameters.

    Args:
      encoder_inputs: A list of 2D Tensors [batch_size x input_size].
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
      dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell in the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "attention_rnn_seq2seq"):
        enc_cell = seq2seq.copy.deepcopy(cell)
        encoder_outputs, enc_state = seq2seq.rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [
            seq2seq.array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
            ]

        attention_states = seq2seq.array_ops.concat(top_states, 1)

        return seq2seq.attention_decoder(decoder_inputs, enc_state, attention_states, cell)