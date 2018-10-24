#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:25:37 2018

@author: nn
"""

class BaseTrainer:
  def __init__(self, trainer_setting):
    self.setting = trainer_setting    
    
  def make_model(self, loader, is_train=True):
    raise NotImplementedError()

  def get_losses(self, inputs, models, loss_params):
    raise NotImplementedError()
    
  def get_trainer(self, inputs, models, losses):
    raise NotImplementedError()

  def get_evaluator(self, inputs, models):
    raise NotImplementedError()

  def initialize_training(self, loader):
    raise NotImplementedError()

  def check_run(self, key, func):
   if key in self.setting:
     func(**self.setting[key])
  
  def check_blank_run(self, key, func):
    params = self.setting[key] if key in self.setting else {}
    func(**params)

  def make_graph(self, loader, is_train):
    inputs, models = self.make_model(loader, is_train=is_train)
    self.inputs = inputs
    self.models = models
    if is_train:
      losses = self.get_losses(inputs, models, self.setting["loss_params"])
      trainers = self.get_trainer(inputs, models, losses)    
      self.losses = losses
      self.trainers = trainers
    else:
      losses = self.get_losses(inputs, models, self.setting["loss_params"])
#      trainers = self.get_trainer(inputs, models, losses)    
      self.evaluator = self.get_evaluator(inputs, models)
      
    self.initialize_training(loader)
    
  def train(self, loader, epochs, batch_size=32):
    raise NotImplementedError()
    
  def evaluate(self, loader, eval_params):
    raise NotImplementedError()
    
