#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 07:10:18 2018

@author: nn
"""

import tensorflow as tf

def get_mlp(feature, n_out, is_train, mlp_params, with_softmax=False,
            var_name="mlp"):  
  hidden_layers = mlp_params["hidden_layers"]
  act = tf.nn.relu
  x = feature
  with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
    for n in hidden_layers:
      x = tf.layers.dense(x, n, activation=act)
    x = tf.layers.dense(x, n_out)
    if with_softmax:
      x = tf.nn.softmax(x)
    return x