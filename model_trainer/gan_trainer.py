#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 06:12:36 2018

@author: nn
"""

from tf_base_trainer import TFBaseTrainer

class GanBaseTrainer(TFBaseTrainer):
  def make_model(self, shape, n_classes, is_train=True):
    raise NotImplementedError()

  def get_losses(self, inputs, models, loss_params):
    raise NotImplementedError()
    
  def get_trainer(self, inputs, models, losses):
    raise NotImplementedError()

  def train(self, loader, epochs, batch_size=32):
    raise NotImplementedError()

