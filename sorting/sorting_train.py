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

"""Trains a model that sorts numbers, keeping loss summaries in tensorboard.

The flag hparam has to be passed as a string of comma separated statements of
the form hparam=value, where the hparam's are any of the listed in the
dictionary DEFAULT_HPARAMS.
See the README.md file for further compilation and running instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sorting_model

flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

flags.DEFINE_string('hparams', '', 'Hyperparameters')
flags.DEFINE_integer('num_iters', 500, 'Number of iterations')
flags.DEFINE_integer(
    'save_summaries_secs', 30,
    'The frequency with which summaries are saved, in seconds.')
flags.DEFINE_integer(
    'save_interval_secs', 30,
    'The frequency with which the model is saved, in seconds.')
flags.DEFINE_string('exp_log_dir', '/tmp/sorting/',
                    'Directory where to write event logs.')
flags.DEFINE_integer('max_to_keep', 1, 'Maximum number of checkpoints to keep')

DEFAULT_HPARAMS = tf.contrib.training.HParams(n_numbers=50,
                                              lr=0.1,
                                              temperature=1.0,
                                              batch_size=10,
                                              prob_inc=1.0,
                                              samples_per_num=5,
                                              n_iter_sinkhorn=10,
                                              n_units=32,
                                              noise_factor=1.0,
                                              optimizer='adam',
                                              keep_prob=1.)


def main(_):

  hparams = DEFAULT_HPARAMS
  hparams.parse(FLAGS.hparams)

  if not gfile.Exists(FLAGS.exp_log_dir):
    gfile.MakeDirs(FLAGS.exp_log_dir)
  tf.reset_default_graph()
  g = tf.Graph()
  model = sorting_model.SortingModel(g, hparams)
  with g.as_default():
    model.set_input()
    model.build_network()
    model.build_l2s_loss()
    model.build_optimizer()
    model.add_summaries_train()

    with tf.Session():
      tf.contrib.slim.learning.train(
          train_op=model.train_op,
          logdir=FLAGS.exp_log_dir,
          global_step=model.global_step,
          saver=tf.train.Saver(max_to_keep=FLAGS.max_to_keep),
          number_of_steps=FLAGS.num_iters,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  tf.app.run(main)
