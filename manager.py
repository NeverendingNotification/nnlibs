#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 08:03:51 2018

@author: nn
"""

from data_loader import basic_loader
from model_trainer import tf_base_trainer

def get_loader(loader_params):
  loader = basic_loader.get_data_loader(loader_params)
  return loader

def get_trainer(trainer_params, loader, is_train=True):
  train_type = trainer_params["train_type"]
  if train_type == "sl":    
    trainer = tf_base_trainer.SLBaseTrainer(trainer_params)
    print(trainer_params)
    print(tf_base_trainer.SLBaseTrainer)
    
  trainer.make_graph(loader, is_train)
  return trainer

def get_runner(runner_params, loader, trainer):
  class Runner:
    def __init__(self, runner_params, loader, trainer):
      self.params = runner_params
      self.loader = loader
      self.trainer = trainer

    def run(self):
      self.trainer.train(loader, self.params["epochs"],
                         self.params["batch_size"])
  return Runner(runner_params, loader, trainer)
      

if __name__ == "__main__":
  loader_params = {}
  loader_params["data_type"] = "mnist"
  loader = get_loader(loader_params)
  print(loader)
  train_loader = loader["train"]
  arr = train_loader.get_images([0], raw=True)  
  import numpy as np
  print(arr.shape, np.min(arr), np.max(arr))
  out = arr[0].astype(np.uint8)
  import cv2
  cv2.imshow("test", out)
  cv2.waitKey(3000)
  