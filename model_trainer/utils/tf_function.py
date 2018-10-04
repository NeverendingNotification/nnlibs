#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 22:46:35 2018

@author: nn
"""

import tensorflow as tf

def step_decay(epoch, initial=0.001, drop=0.5,
               epochs_drop=10.0):
   val = initial * tf.pow(drop, tf.cast(tf.floordiv(epoch, epochs_drop),
                                        tf.float32))
   return val

def exp_decay(epoch, initial, decay_rate=0.1):
   val = initial * tf.exp(-decay_rate * tf.cast(epoch, tf.float32))
   return val
 