#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 06:10:45 2018

@author: nn
"""

import tensorflow as tf

from . import conv_layers
from . import feature_extractor
from . import mlp

def get_neural_net(input_, is_train, n_classes, conv_params,
                   feature_params, mlp_params, var_name="network"):
  with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
    conv_last = conv_layers.get_conv_layers(input_, is_train, conv_params)
    feature = feature_extractor.get_feature_extractor(conv_last, feature_params)
    logits = mlp.get_mlp(feature, n_classes, is_train, mlp_params)
    return conv_last, feature, logits
    


def get_network(model_params):
  shape = model_params["shape"]
  with tf.name_scope("inputs"):
    in_imgs = tf.placeholder(tf.float32, shape=(None,) + shape)
    is_train = tf.placeholder(tf.bool)
    trg_imgs = tf.placeholder(tf.int32, shape=(None,))
  inputs = {"input":in_imgs, "is_train":is_train, "target":trg_imgs}
  
  conv_last, feature, logits = get_neural_net(in_imgs, is_train,
                                              model_params["n_classes"],
                                              model_params["conv_params"],
                                              model_params["feature_layer"],
                                              model_params["mlp_params"]
                                              )
  
  pred = tf.nn.softmax(logits)
  models = {"conv_output":conv_last, "feature":feature,
            "logits":logits, "prediction":pred}
  return inputs, models

