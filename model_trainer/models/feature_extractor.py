#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 07:02:07 2018

@author: nn
"""

import tensorflow as tf

def global_average_pooling(conv_output):
  with tf.name_scope("GAP"):
    out = tf.reduce_mean(conv_output, axis=[1, 2])
  return out


def get_feature_extractor(conv_output, feature_type):
  if feature_type == "gap":
    return global_average_pooling(conv_output)
  elif feature_type == "flatten":
    return tf.layers.flatten(conv_output, name="flatten")
  else:
    raise NotImplementedError()
