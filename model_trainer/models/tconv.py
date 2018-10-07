#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:18:55 2018

@author: nn
"""

import tensorflow as tf


def get_tconv(feature, out_shape, is_train, tconv_params, with_softmax=False,
            var_name="tconv"):
  act = tf.nn.relu
  h, w, c = out_shape
  n_up = tconv_params["n_up"]
  i_channel = tconv_params["i_channel"]
  
  h0 = h // 2 ** n_up
  w0 = w // 2 ** n_up
  with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
    first_layer = tf.layers.dense(feature, h0 * w0 * i_channel, activation=act)
    x = tf.reshape(first_layer, [-1, h0, w0, i_channel])
    for n in range(n_up):
      n_channel = i_channel // 2**n
      x = tf.layers.conv2d(x, n_channel, 3, padding="same", activation=act)
      x = tf.layers.conv2d_transpose(x, n_channel, 3, strides=2, padding="same",
                                    activation=act)
    x = tf.layers.conv2d(x, c, 3, padding="same")
    return x