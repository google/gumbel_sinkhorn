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


"""Model class for sorting numbers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import optimizer
import sinkhorn_ops


class SortingModel(object):
  """Constructs the graph of tensors to learn to sort numbers."""

  def __init__(self, graph, hparams):
    self.graph = graph
    self.hparams = hparams
    self.batch_size = hparams.batch_size
    self.n_numbers = hparams.n_numbers
    self.samples_per_num = hparams.samples_per_num
    self.n_iter_sinkhorn = hparams.n_iter_sinkhorn
    self.noise_factor = hparams.noise_factor
    self.prob_inc = hparams.prob_inc
    self.optimizer = hparams.optimizer
    self.n_units = hparams.n_units

  def set_input(self):
    with self.graph.as_default():
      (self._ordered, self._random,
       self._hard_perms) = sinkhorn_ops.sample_uniform_and_order(
           self.batch_size, self.n_numbers, self.prob_inc)
      # tiled variables, to compare to many permutations
      self._ordered_tiled = tf.tile(self._ordered, [self.samples_per_num, 1])
      self._random_tiled = tf.tile(self._random, [self.samples_per_num, 1])

  def build_network(self):
    """The most important part, where the neural network is built."""

    def _create_log_alpha(self):
      """Creates the variable log_alpha, through NN processing of input."""

      with tf.variable_scope("model_params"):
        # each number is processed with the same network, so data is reshaped
        # so that numbers occupy the 'batch' position.
        random_flattened = tf.reshape(self._random, [-1, 1])
        # net: output of the first neural network that connects numbers to a
        # 'latent' representation.
        net = dropout(fc(random_flattened, self.n_units), self.keep_prob)
        # now those latent representation is connected to rows of the matrix
        # log_alpha.
        processed = dropout(
            fc(net, self.n_numbers, activation_fn=None), self.keep_prob)

        # the matrix log_alpha is created by concatenation of the rows
        # corresponding to different numbers.
        return  tf.reshape(processed, [-1, self.n_numbers, self.n_numbers])

    with self.graph.as_default():

      self.keep_prob = tf.constant(self.hparams.keep_prob, dtype=tf.float32)
      self.temperature = tf.constant(self.hparams.temperature, dtype=tf.float32)
      self._global_step = tf.Variable(0, trainable=False)
      fc = tf.contrib.layers.fully_connected
      dropout = tf.contrib.layers.dropout

      self._log_alpha = _create_log_alpha(self)
      # Now, we sample using gumbel_sinkhorn from the
      # constructed matrix log_alpha.
      (self._soft_perms_inf,
       self._log_alpha_w_noise) = sinkhorn_ops.gumbel_sinkhorn(
           self._log_alpha, self.temperature, self.samples_per_num,
           self.noise_factor, self.n_iter_sinkhorn, squeeze=False)

  def build_initializer(self):
    with self.graph.as_default():
      tf.initialize_all_variables()

  def build_l2s_loss(self):
    """Builds loss tensor with soft permutations, for training."""
    with self.graph.as_default():
      inv_soft_perms = tf.transpose(self._soft_perms_inf, [0, 1, 3, 2])
      inv_soft_perms_flat = tf.reshape(
          tf.transpose(inv_soft_perms, [1, 0, 2, 3]),
          [-1, self.n_numbers, self.n_numbers])
      ordered_tiled = tf.reshape(self._ordered_tiled, [-1, self.n_numbers, 1])
      random_tiled = tf.reshape(self._random_tiled, [-1, self.n_numbers, 1])
      # squared l2 loss
      self._l2s_diff = tf.reduce_mean(
          tf.square(
              ordered_tiled - tf.matmul(inv_soft_perms_flat, random_tiled)))

  def build_hard_losses(self):
    """Losses based on hard reconstruction. Only for evaluation.

    Doubly stochastic matrices are rounded with
    the matching function.
    """

    log_alpha_w_noise_flat = tf.reshape(tf.transpose(self._log_alpha_w_noise,
                                                     [1, 0, 2, 3]),
                                        [-1, self.n_numbers, self.n_numbers])

    hard_perms_inf = sinkhorn_ops.matching(log_alpha_w_noise_flat)
    inverse_hard_perms_inf = sinkhorn_ops.invert_listperm(hard_perms_inf)
    hard_perms_tiled = tf.tile(self._hard_perms,
                               [self.samples_per_num, 1])

    # The 3D output of permute_batch_split must be squeezed
    self._ordered_inf_tiled = tf.reshape(
        sinkhorn_ops.permute_batch_split(
            self._random_tiled, inverse_hard_perms_inf),
        [-1, self.n_numbers])

    self._l1_diff = tf.reduce_mean(
        tf.abs(self._ordered_tiled - self._ordered_inf_tiled))
    self._l2sh_diff = tf.reduce_mean(
        tf.square(self._ordered_tiled - self._ordered_inf_tiled))
    diff_perms = tf.cast(
        tf.abs(hard_perms_tiled - inverse_hard_perms_inf), tf.float32)
    self._prop_wrong = -tf.reduce_mean(tf.sign(-diff_perms))
    self._prop_any_wrong = -tf.reduce_mean(
        tf.sign(-tf.reduce_sum(diff_perms, 1)))
    self._kendall_tau = tf.reduce_mean(
        sinkhorn_ops.kendall_tau(hard_perms_tiled, inverse_hard_perms_inf))

  def build_optimizer(self):
    with self.graph.as_default():
      opt = optimizer.set_optimizer(self.hparams.optimizer,
                                    self.hparams.lr, opt_eps=1e-8)
      self._train_op = tf.contrib.training.create_train_op(
          self._l2s_diff, opt, global_step=self._global_step)

  def build_train_ops(self):
    with self.graph.as_default():
      self._vars = tf.trainable_variables()
      self._train_op = tf.train.AdamOptimizer(
          self.hparams.lr).minimize(self._l2s_diff,
                                    var_list=self._vars,
                                    global_step=self._global_step)

  def add_summaries_train(self):
    """Adds necessary summaries which will be computed during training."""
    with tf.name_scope("Training"):
      with self.graph.as_default():
        tf.summary.scalar("Total_l2_squared_loss", self._l2s_diff)

  def add_summaries_eval(self):
    """Adds necessary summaries which will be computed during evaluation."""
    with tf.name_scope("Evaluation"):
      with self.graph.as_default():
        tf.summary.scalar("L1_diff", self._l1_diff)
        tf.summary.scalar("L2_squared_diff", self._l2sh_diff)
        tf.summary.scalar("Proportion_wrong", self._prop_wrong)
        tf.summary.scalar("Proportion_where_any_is_wrong",
                          self._prop_any_wrong)
        tf.summary.scalar("Kendall's_tau",
                          self._kendall_tau)

  def get_eval_measures(self):
    """Getter method for evaluation measures."""
    return (self._l1_diff, self._l2sh_diff, self._kendall_tau,
            self._prop_wrong, self._prop_any_wrong)

  @property
  def train_op(self):
    return self._train_op

  @property
  def ordered_inf(self):
    return tf.transpose(
        tf.reshape(self._ordered_inf_tiled,
                   [self.samples_per_num, -1, self.n_numbers]), [1, 0, 2])

  @property
  def ordered(self):
    return tf.transpose(
        tf.reshape(self._ordered_tiled,
                   [self.samples_per_num, -1, self.n_numbers]), [1, 0, 2])

  @property
  def random(self):
    return tf.transpose(
        tf.reshape(self._random_tiled,
                   [self.samples_per_num, -1, self.n_numbers]), [1, 0, 2])

  @property
  def global_step(self):
    return self._global_step

