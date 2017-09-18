# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Library with optimization definitions and functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def set_optimizer(optimizer, lr, opt_eps=1.0, opt_momentum=0.9, rms_decay=0.9,
                  adam_beta1=0.9, adam_beta2=0.999):
  """Sets optimizer optimizer op.

  Args:
   optimizer: A string (sgd, momentum, adagrad, adam, rmsprop).
   lr: learning rate, a float.
   opt_eps: Optimizer epsilon (for ADAM and RMSprop).
   opt_momentum: Optimizer momentum. Common for Momentum and RMSProp.
   rms_decay: RMSProp decay parameter.
   adam_beta1: beta_1 parameter for ADAM.
   adam_beta2: beta_2 parameter for ADAM.
  Returns:
    opt, the optimizer op.

  """

  if optimizer == "sgd":
    opt = tf.train.GradientDescentOptimizer(lr)
  elif optimizer == "momentum":
    opt = tf.train.MomentumOptimizer(lr, opt_momentum)
  elif optimizer == "adagrad":
    opt = tf.train.AdagradOptimizer(lr)
  elif optimizer == "adam":
    opt = tf.train.AdamOptimizer(lr, beta1=adam_beta1, beta2=adam_beta2,
                                 epsilon=opt_eps)
  elif optimizer == "rmsprop":
    opt = tf.train.RMSPropOptimizer(lr, rms_decay, opt_momentum, opt_eps)
  return opt
