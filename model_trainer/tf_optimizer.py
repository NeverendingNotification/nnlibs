#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:03:35 2018

@author: nn
"""

import tensorflow as tf
from .utils import tf_function


def get_optimizer(epoch, opt_params):
  lrate_params = opt_params["lrate_params"]
  opt_type = opt_params["opt_type"]
  lr = lrate_params["l_rate"]
  if "decay_type" not in lrate_params:
    lrate = tf.constant(lr)
  elif lrate_params["decay_type"] == "step":
    lrate = tf_function.step_decay(epoch, initial=lr,
                                   drop=opt_params["decay_factor"],
                                   epochs_drop=lrate_params["decay_epoch"])
  elif lrate_params["decay_type"] == "exp":
    lrate = tf_function.exp_decay(epoch, lr,
                                  decay_rate=lrate_params["decay_rate"])
  if opt_type == "adam":
    opt = tf.train.AdamOptimizer(learning_rate=lrate)
  elif opt_type == "sgd":
    opt = tf.train.GradientDescentOptimizer(lrate)
  elif opt_type == "rms":
    opt = tf.train.RMSPropOptimizer(lrate)
  return opt, lrate
