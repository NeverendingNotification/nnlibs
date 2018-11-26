#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 06:19:14 2018

@author: nn
"""

import tensorflow as tf

def data_augmentation(img):
  with tf.name_scope("data_augmentation"):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
  return img

