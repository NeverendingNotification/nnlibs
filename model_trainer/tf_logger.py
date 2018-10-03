#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:07:19 2018

@author: nn
"""

import os

from . import tf_metrics

class BaseLogger:
  def __init__(self, log_dir=None, metrics={}, metric_period=1):
    if log_dir is not None:
      if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    self.log_dir = log_dir
    self.metrics = metrics
  def start_epoch(self, trainer, loader, epoch):
    self.loss = 0

  def log_batch(self, batch):
    self.loss += batch["loss"]
  
  
  def end_epoch(self, trainer, loader, epoch):
    out = tf_metrics.get_metrics_classifier(loader, trainer, 
                                      metrics=self.metrics)
    key = ", ".join(["{} : {}".format(metric, out[metric]) for metric in self.metrics])
    print("Epoch : {}, loss = {}, {}".format(epoch, self.loss, key))
  
  def log_end(self, trainer, loader):
    pass