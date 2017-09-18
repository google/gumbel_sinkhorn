"""Evaluates a sorting model, adding scalar summaries to tensorboard.

It also outputs evaluation metrics as numpy scalars.
The flag hparam has to be passed as a string of comma separated statements of
the form hparam=value, where the hparam's are any of the listed in the
dictionary DEFAULT_HPARAMS.
See the README.md file for further compilation and running instructions.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time

import tensorflow as tf

import sorting_model

gfile = tf.gfile
flags = tf.app.flags
flags.DEFINE_string("hparams", "", "Hyperparameters")
flags.DEFINE_string("batch_transform_type",
                    None, "Options: None, WeakMix, StrongMix")
flags.DEFINE_boolean("evaluate_all",
                     True, "If false evaluate only the last checkpoint")
flags.DEFINE_string("exp_log_dir", "/tmp/sorting/",
                    "Directory where to write event logs.")
flags.DEFINE_boolean("eval_once", True,
                     "If true, only evaluate the model once")
flags.DEFINE_integer("sec_sleep", 30,
                     "If no new checkpoint, sleep for some seconds")
flags.DEFINE_integer("secs_run_for", 10000,
                     "Maximum time (seconds) the program will be active")

FLAGS = tf.app.flags.FLAGS
DEFAULT_HPARAMS = tf.contrib.training.HParams(n_numbers=50,
                                              lr=0.1,
                                              temperature=1.0,
                                              batch_size=50,
                                              prob_inc=1.0,
                                              samples_per_num=5,
                                              n_iter_sinkhorn=10,
                                              n_units=32,
                                              noise_factor=0.0,
                                              optimizer="adam",
                                              keep_prob=1.)


def log(s):
  tf.logging.info(s)
  print(s)


def wait_for_new_checkpoint(saver, sess, logdir, global_step,
                            last_step_evaluated, sleep_secs):
  while True:
    if restore_checkpoint_if_exists(saver, sess, logdir):
      step = sess.run(global_step)
      if step <= last_step_evaluated:
        log("Found old checkpoint, sleeping %ds" % sleep_secs)
        time.sleep(sleep_secs)
      else:
        return step
    else:
      log("Checkpoint not found in %s,"
          "sleeping for %ds" % (logdir, sleep_secs))
      time.sleep(sleep_secs)


def restore_checkpoint_if_exists(saver, sess, logdir):
  ckpt = tf.train.get_checkpoint_state(logdir)
  if ckpt:
    log("Restoring checking point from %s" % ckpt.model_checkpoint_path)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    full_checkpoint_path = os.path.join(logdir, ckpt_name)
    saver.restore(sess, full_checkpoint_path)
    return True
  return False


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

def main(_):
  time_start = time.time()
  hparams = DEFAULT_HPARAMS
  hparams.parse(FLAGS.hparams)

  if not gfile.Exists(FLAGS.exp_log_dir):
    gfile.MakeDirs(FLAGS.exp_log_dir)

  g = tf.Graph()
  model = sorting_model.SortingModel(g, hparams)

  with model.graph.as_default():
    model.set_input()
    model.build_network()
    model.build_hard_losses()
    model.add_summaries_eval()
    summaries_eval = tf.summary.merge_all()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.exp_log_dir, model.graph)
    last_step_evaluated = -1
    with tf.Session() as session:
      while time.time() - time_start < FLAGS.secs_run_for:
        wait_for_new_checkpoint(
            saver, session, FLAGS.exp_log_dir,
            model.global_step, last_step_evaluated, FLAGS.sec_sleep)
        (summaries, step, eval_measures) = session.run([
            summaries_eval, model.global_step, model.get_eval_measures()])
        (l1_diff, l2sh_diff, kendall_tau,
         prop_wrong, prop_any_wrong) = eval_measures
        log("Frequency of mistakes was %s" % prop_wrong)
        log("Frequency of series with at least an error was %s" %
            prop_any_wrong)
        log("Kendall's tau was %s" % kendall_tau)
        log(("Mean L2 squared difference between true and inferred series "
             " was %s") % l2sh_diff)
        log("Mean L1 difference between true and inferred series was %s"
            % l1_diff)
        writer.add_summary(summaries, global_step=step)
        last_step_evaluated = step
        if FLAGS.eval_once is True:
          break

if __name__ == "__main__":
  tf.app.run()
