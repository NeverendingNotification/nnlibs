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
    self.all_metrics = {}
    self.is_out = False
    self.threshold = 0.5
    
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
    if len(self.all_metrics) > 0:
      import pandas as pd
      df = pd.DataFrame(index=self.all_metrics["epoch"])
      for metric in self.metrics:
        df[metric] = self.all_metrics[metric]
      df.to_csv(os.path.join(self.log_dir, "metrics.csv"))
      df.plot()
      import matplotlib.pyplot as plt
      plt.savefig(os.path.join(self.log_dir, "plot.png"))
      
  def is_outmodel(self):
    return False
  
class SlLogger(BaseLogger):
  def __init__(self, **params):
    super().__init__(**params)
    self.max_iou = 0.0

  
  def end_epoch(self, trainer, loader, epoch):
    if "epoch" not in self.all_metrics:
      self.all_metrics["epoch"] = []
    self.all_metrics["epoch"].append(epoch)
    out = tf_metrics.get_metrics_classifier(loader, trainer, 
                                      metrics=self.metrics)
    for metric in self.metrics:
      if metric not in self.all_metrics:
        self.all_metrics[metric] = []
      self.all_metrics[metric].append(out[metric])    
    loss_key = self.get_loss_str()
    key = ", ".join(["{} : {:.04f}".format(metric, out[metric]) for metric in self.metrics])
    print("Epoch : {}, {}, {}".format(epoch, loss_key, key), self.max_iou)
    self.is_out = False
    if  out["max_iou"] > self.max_iou:
      self.max_iou = out["max_iou"]
      if self.max_iou > 0.1:
        self.is_out = True
        self.threshold = out["max_iou_threshold"]
        print("IOU THRESHOLD", out["max_iou_threshold"])
    
    
  def is_outmodel(self):
    if self.is_out:
      self.is_out = False
      return True
    else:
      return False
    
  
    
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
  
  