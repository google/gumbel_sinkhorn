# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for sinkhorn_ops library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import google3
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test
import sinkhorn_ops


class SinkhornTest(test.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(0)
    tf.set_random_seed(1)

  def test_approximately_stochastic(self):
    with self.test_session(use_gpu=True):
      for dims in [2, 5, 10]:
        for batch_size in [1, 2, 10]:
          log_alpha = self.rng.randn(batch_size, dims, dims)
          result = sinkhorn_ops.sinkhorn(log_alpha)

          self.assertAllClose(np.sum(result.eval(), 1),
                              np.tile([1.0], (batch_size, dims)),
                              atol=1e-3)
          self.assertAllClose(np.sum(result.eval(), 2),
                              np.tile([1.0], (batch_size, dims)),
                              atol=1e-3)

  def test_equivalence_gumbel_sinkhorn_and_sinkhorn(self):
    """Tests the equivalence between sinkhorn and gumbel_sinhorn in a case.

    When noise_factor = 0.0 the output of gumbel_sinkhorn should be the same
    as the output of sinkhorn, 'modulo' possible many repetitions of the same
    matrix given by gumbel_sinkhorn.
    """
    with self.test_session(use_gpu=True):
      batch_size = 10
      dims = 5
      n_samples = 20
      temp = 1.0
      noise_factor = 0.0
      log_alpha = self.rng.randn(batch_size, dims, dims)
      result_sinkhorn = sinkhorn_ops.sinkhorn(log_alpha)
      result_sinkhorn_reshaped = tf.reshape(
          result_sinkhorn, [batch_size, 1, dims, dims])
      result_sinkhorn_tiled = tf.tile(
          result_sinkhorn_reshaped, [1, n_samples, 1, 1])
      result_gumbel_sinkhorn, _ = sinkhorn_ops.gumbel_sinkhorn(
          log_alpha, temp, n_samples, noise_factor)

      self.assertAllEqual(result_gumbel_sinkhorn.eval(),
                          result_sinkhorn_tiled.eval())

  def test_gumbel_sinkhorn_high_temperature(self):
    """At very high temperatures, the resulting matrix approaches the uniform.
    """
    n_samples = 1
    temp = 100000.0

    with self.test_session(use_gpu=True):

      for dims in [2, 5, 10]:
        for batch_size in [1, 2, 10]:
          for noise_factor in [1.0, 5.0]:
            log_alpha = tf.cast(self.rng.randn(batch_size, dims, dims),
                                dtype=tf.float32)

            result_gumbel_sinkhorn, _ = sinkhorn_ops.gumbel_sinkhorn(
                log_alpha, temp, n_samples, noise_factor, squeeze=True)
            uniform = np.ones((batch_size, dims, dims)) / dims
            self.assertAllClose(uniform, result_gumbel_sinkhorn.eval(),
                                atol=1e-3)

  def test_matching(self):
    """The solution of the matching for the identity matrix is range(N).
    """
    with self.test_session(use_gpu=True):
      dims = 10
      identity = np.eye(dims)
      result_matching = sinkhorn_ops.matching(identity)
      self.assertAllEqual(result_matching.eval(),
                          np.reshape(range(dims), [1, dims]))

  def test_perm_inverse(self):
    """The product of a permutation and its inverse is the identity."""

    with self.test_session(use_gpu=True):
      dims = 10
      permutation = np.reshape(self.rng.permutation(dims), [1, -1])
      permutation_matrix = sinkhorn_ops.listperm2matperm(permutation)
      inverse = sinkhorn_ops.invert_listperm(permutation)
      inverse_matrix = sinkhorn_ops.listperm2matperm(inverse)
      prod = tf.matmul(permutation_matrix, inverse_matrix)

      self.assertAllEqual(prod.eval(),
                          np.reshape(np.eye(dims), [1, dims, dims]))

  def test_listperm2matperm(self):
    """The matrix form of the permutation range(N) is the identity."""

    with self.test_session(use_gpu=True):
      dims = 10
      permutation_list = np.reshape(np.arange(dims), [1, -1])
      permutation_matrix = sinkhorn_ops.listperm2matperm(permutation_list)
      self.assertAllEqual(permutation_matrix.eval(),
                          np.reshape(np.eye(dims), [1, dims, dims]))

  def test_matperm2listperm(self):
    """The list form of the matrix permutation identity is range(N)."""

    with self.test_session(use_gpu=True):
      dims = 10
      permutation_matrix = np.eye(dims)
      permutation_list = sinkhorn_ops.matperm2listperm(permutation_matrix)
      self.assertAllEqual(permutation_list.eval(),
                          np.reshape(np.arange(dims), [1, dims]))

  def test_sample_uniform_and_order(self):
    """Ordered numbers form indeed an increasing sequence."""
    n_lists = 1
    n_numbers = 10
    prob_inc = 1.0
    with self.test_session(use_gpu=True):
      ordered, _, _ = sinkhorn_ops.sample_uniform_and_order(n_lists,
                                                            n_numbers,
                                                            prob_inc)
      self.assertTrue(np.min(np.diff(ordered.eval())) > 0)

  def test_sample_permutations(self):
    """What is being sampled are indeed permutations of range(N)."""
    n_permutations = 10
    n_objects = 5

    with self.test_session(use_gpu=True):
      permutations = sinkhorn_ops.sample_permutations(n_permutations, n_objects)
      tiled_range = np.tile(np.reshape(
          np.arange(n_objects), [1, n_objects]), [n_permutations, 1])
      self.assertAllEqual(np.sort(permutations.eval()), tiled_range)

if __name__ == '__main__':
  tf.test.main()
