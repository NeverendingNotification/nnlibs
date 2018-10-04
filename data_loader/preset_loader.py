#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 10:34:21 2018

@author: nn
"""

import tensorflow as tf

def load_data(data_type):
  if data_type == "mnist":
    tr, te = tf.keras.datasets.mnist.load_data()
    train = tr[0].reshape([-1, 28, 28, 1])
    test = te[0].reshape([-1, 28, 28, 1])
    return (train, tr[1]), (test, te[1]), 10
  elif data_type == "cifar10":
    tr, te = tf.keras.datasets.cifar10.load_data()
    train = tr[0].reshape([-1, 32, 32, 3])
    test = te[0].reshape([-1, 32, 32, 3])
    return (train, tr[1].reshape([-1])), (test, te[1].reshape([-1])), 10
  else:
    raise TypeError()

