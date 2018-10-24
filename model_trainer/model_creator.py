#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 06:10:45 2018

@author: nn
"""

import tensorflow as tf

from .models import cnn

def make_model(model_arch, is_train=True):
  arch_type = model_arch["network_type"]  
  if arch_type == "cnn":
    inputs, models = cnn.get_network(model_arch)
    return inputs, models
  elif arch_type == "cnn_ae":
    inputs, models = cnn.get_ae_network(model_arch)
    return inputs, models    
  elif arch_type == "gan":
    inputs, models = cnn.get_gan_network(model_arch)
    return inputs, models    
  else:
    raise NotImplementedError()