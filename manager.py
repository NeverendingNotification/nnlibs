#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 08:03:51 2018

@author: nn
"""

from data_loader import basic_loader
from model_trainer import tf_base_trainer
from model_trainer import gan_trainer

def get_loader(loader_params, mode="train", arc_type="sl"):
  loader = basic_loader.get_data_loader(loader_params, mode=mode)
  return loader

def get_trainer(trainer_params, loader, mode="train", arc_type="sl",
                out_root=None):
  is_train = mode == "train"
  if out_root is not None:
    trainer_params["out_root"] = out_root
  trainer_params["arc_type"] = arc_type
  if arc_type == "sl":    
    trainer = tf_base_trainer.SLBaseTrainer(trainer_params)
    print(trainer_params)
    print(tf_base_trainer.SLBaseTrainer)
  elif arc_type == "ae":
    trainer = tf_base_trainer.AEBaseTrainer(trainer_params)
    print(trainer_params)
    print(tf_base_trainer.SLBaseTrainer)    
  elif arc_type == "gan":
    if "sub_type" not in trainer_params:
      trainer = gan_trainer.GanBaseTrainer(trainer_params)
    else:
      sub_type = trainer_params["sub_type"]
      if sub_type == "wgan":
        trainer = gan_trainer.WGanBaseTrainer(trainer_params)
      elif sub_type == "wgangp":
        trainer = gan_trainer.WGanGpTrainer(trainer_params)
  trainer.make_graph(loader, is_train)
  return trainer

def get_runner(runner_params, loader, trainer, mode, arc_type="sl"):
  class Runner:
    def __init__(self, runner_params, loader, trainer):
      self.params = runner_params
      self.loader = loader
      self.trainer = trainer

    def run(self):
      if mode == "train":
        self.trainer.train(loader, self.params["epochs"],
                         self.params["batch_size"])
      elif mode == "eval":
        self.trainer.evaluate(loader, self.params["eval_params"])
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
  