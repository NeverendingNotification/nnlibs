#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:07:19 2018

@author: nn
"""

from collections import OrderedDict

import os

import cv2

from . import tf_metrics

def get_logger(arc_type, logger_params):
  if arc_type == "sl":
    logger = SlLogger(**logger_params)
  elif arc_type == "ae":
    logger = AELogger(**logger_params)
  elif arc_type == "gan":
    logger = GanLogger(**logger_params)
  return logger

class BaseLogger:
  def __init__(self, log_dir=None, out_root=None, metrics={}, metric_period=1,
               sample_dirname="sample"):
    if out_root is not None:
      log_dir = os.path.join(out_root, log_dir)
    if log_dir is not None:
      if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    self.log_dir = log_dir
    self.metrics = metrics
    self.sample = sample_dirname
  def start_epoch(self, trainer, loader, epoch):
    self.losses = OrderedDict()

  def log_batch(self, batch, loss_keys=["loss"]):
    for key in loss_keys:
      if key not in batch:
        continue
      if key not in self.losses:
        self.losses[key] = 0.0
      self.losses[key] += batch[key]
  
  
  def end_epoch(self, trainer, loader, epoch):
    raise NotImplementedError()
  
  def get_loss_str(self):
    key = ", ".join(["{} : {:.04f}".format(k, v) for k, v in self.losses.items()])
    return key
  
  def log_end(self, trainer, loader):
    pass
  
class SlLogger(BaseLogger):
  def end_epoch(self, trainer, loader, epoch):
    out = tf_metrics.get_metrics_classifier(loader, trainer, 
                                      metrics=self.metrics)
    loss_key = self.get_loss_str()
    key = ", ".join(["{} : {}".format(metric, out[metric]) for metric in self.metrics])
    print("Epoch : {}, {}, {}".format(epoch, loss_key, key))
    
class AELogger(BaseLogger):
  def end_epoch(self, trainer, loader, epoch):
    out, images = tf_metrics.get_metrics_generator(loader, trainer, 
                                      metrics=self.metrics)
    o_dir = os.path.join(self.log_dir, self.sample)
    if not os.path.isdir(o_dir):
      os.makedirs(o_dir)
      
    for i, image in enumerate(images):
      cv2.imwrite(os.path.join(o_dir, "{:05d}_{:04d}.png".format(epoch, i)), image)

    loss_key = self.get_loss_str()
    key = ", ".join(["{} : {}".format(metric, out[metric]) for metric in self.metrics])
    print("Epoch : {}, {}, {}".format(epoch, loss_key, key))
  
class GanLogger(BaseLogger):
  def end_epoch(self, trainer, loader, epoch):
    out, images = tf_metrics.get_metrics_generator(loader, trainer, 
                                      metrics=self.metrics)
    o_dir = os.path.join(self.log_dir, self.sample)
    if not os.path.isdir(o_dir):
      os.makedirs(o_dir)
      
    for i, image in enumerate(images):
      cv2.imwrite(os.path.join(o_dir, "{:05d}_{:04d}.png".format(epoch, i)), image)

    loss_key = self.get_loss_str()
    key = ", ".join(["{} : {}".format(metric, out[metric]) for metric in self.metrics])
    print("Epoch : {}, {}, {}".format(epoch, loss_key, key))
  
  