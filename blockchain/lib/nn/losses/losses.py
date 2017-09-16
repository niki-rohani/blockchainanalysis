
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.ops.losses import losses as l

import tensorflow as tf


def biased_mean_squared_error(labels, predictions, bias=0., weights=1.0, scope=None,
                       loss_collection=ops.GraphKeys.LOSSES):
        """
        Compute a weighted biased error. The bias is a int that represent how much the prediction sign is important.
        If label-prediction is the same sign of bias,
        the loss will be (1+|bias|)*(label-prediction)**2, (label-prediction)**2 if not
        if bias = 0 then
        both sign will have same weight
        :param labels:
        :param predictions:
        :param weights:
        :param scope:
        :param loss_collection:
        :return:
        """
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = math_ops.subtract(predictions, labels)
        if bias != 0:
            sign = math_ops.sign(losses)
            if bias > 0:
                sign_ = 1
            else:
                sign_ = -1
            tf.where(sign == sign_, losses*(1+bias*sign_), losses)

        losses = math_ops.square(losses)
        return l.compute_weighted_loss(losses, weights, scope, loss_collection)
