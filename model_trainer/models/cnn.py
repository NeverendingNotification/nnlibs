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
from . import tconv

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


def get_ae_net(in_img, is_train,
               shape,
               conv_params,
               feat_params,
               tconv_params, var_name="network"):
  with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
    conv_last = conv_layers.get_conv_layers(in_img, is_train, conv_params)
    feature = feature_extractor.get_feature_extractor(conv_last, feat_params)
    logits = tconv.get_tconv(feature, shape, is_train, tconv_params)
    return conv_last, feature, logits
    

def get_ae_network(model_params):
  shape = model_params["shape"]
  with tf.name_scope("inputs"):
    in_imgs = tf.placeholder(tf.float32, shape=(None,) + shape)
    is_train = tf.placeholder(tf.bool)
    trg_imgs = tf.placeholder(tf.float32, shape=(None,) + shape)
  inputs = {"input":in_imgs, "is_train":is_train, "target":trg_imgs}
  
  conv_last, feature, logits = get_ae_net(in_imgs, is_train,
                                          shape,
                                          model_params["conv_params"],
                                          model_params["feature_layer"],
                                          model_params["tconv_params"]
                                          )
  
  pred = tf.nn.sigmoid(logits)
  models = {"conv_output":conv_last, "feature":feature,
            "logits":logits, "prediction":pred}
  return inputs, models


def get_discriminator(img, is_train, discrim_params, var_name="discrimator"):
  with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
    conv = conv_layers.get_conv_layers(img, is_train, discrim_params["conv_params"])
    feature = feature_extractor.get_feature_extractor(conv,
                                                      discrim_params["feature_type"])
    logits = mlp.get_mlp(feature, 1, is_train,
                         discrim_params["mlp_params"])
    return logits

def get_generator(noise, shape, is_train, gen_params, var_name="generator",
                  act=None):
  with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
        logits = tconv.get_tconv(noise, shape, is_train,
                                 gen_params["tconv_params"])
        if act is not None:
          logits = act(logits)
        return logits


def get_gan_network(model_params):
  shape = model_params["shape"]
  hidden_dim = model_params["hidden_dim"]
  with tf.name_scope("inputs"):
    in_imgs = tf.placeholder(tf.float32, shape=(None,) + shape)
    is_train = tf.placeholder(tf.bool)
    noise = tf.placeholder(tf.float32, shape=(None, hidden_dim))
  inputs = {"input":in_imgs, "is_train":is_train, "noise":noise}
  
  dis_name = "discriminator"
  gen_name = "generator"
  
  gen = get_generator(noise, shape, is_train,
                      model_params["gen_params"],
                      var_name=gen_name, act=tf.nn.sigmoid)
  fake_dis = get_discriminator(gen, is_train,
                               model_params["discrim_params"],
                               var_name=dis_name)
  real_dis = get_discriminator(in_imgs, is_train,
                               model_params["discrim_params"],
                               var_name=dis_name)
  
  tvars = tf.trainable_variables()
  gen_vars = [var for var in tvars if var.name.startswith(gen_name)]
  dis_vars = [var for var in tvars if var.name.startswith(dis_name)]
  
  models = {"fake_dis":fake_dis, "real_dis":real_dis, "generator":gen, 
            "gen_vars":gen_vars, "dis_vars":dis_vars}
  return inputs, models

