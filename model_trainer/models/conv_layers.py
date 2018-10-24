#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 07:09:54 2018

@author: nn
"""

import tensorflow as tf


def get_conv_layers(in_img, is_train, conv_layer_params,
                    var_name="conv"):
  
  layer_type = conv_layer_params["layer_type"]
  n_dim = conv_layer_params["n_dim"]
  act = tf.nn.relu
  with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
    if layer_type == "small":
      x = tf.layers.conv2d(in_img, n_dim // 4, 3, padding="same", activation=act)
      x = tf.layers.conv2d(x, n_dim // 4, 3, padding="same", activation=act)
      x = tf.layers.max_pooling2d(x, 2, 2)
      
      x = tf.layers.conv2d(x, n_dim // 2, 3, padding="same", activation=act)
      x = tf.layers.conv2d(x, n_dim // 2,  3, padding="same", activation=act)
      x = tf.layers.max_pooling2d(x, 2, 2)

      x = tf.layers.conv2d(x, n_dim, 3, padding="same", activation=act)
      x = tf.layers.conv2d(x, n_dim, 3, padding="same", activation=act)
      return x
    elif layer_type == "tiny":
      x = tf.layers.conv2d(in_img, n_dim // 4, 3, padding="same", activation=act)
      x = tf.layers.max_pooling2d(x, 2, 2)
      
      x = tf.layers.conv2d(x, n_dim // 2, 3, padding="same", activation=act)
      x = tf.layers.max_pooling2d(x, 2, 2)

      x = tf.layers.conv2d(x, n_dim, 3, padding="same", activation=act)
      return x
    