#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:37:33 2018

@author: nn
"""

import os

import tensorflow as tf

from .basic_trainer import BaseTrainer
from . import model_creator
from . import tf_logger

class TFBaseTrainer(BaseTrainer):
  def __init__(self, trainer_setting):
    super().__init__(trainer_setting)

  
  def initialize_training(self, loader):
    self.sess = tf.Session()
    saver_setting = self.setting["saver_setting"] \
                    if "saver_setting" in self.setting \
                    else {}
    self.set_saver(saver_setting)
    if "load_modeldir" in self.setting:
      self.saver.restore(self.sess, self.setting["load_model_dir"])
    else:
      self.sess.run(tf.global_variables_initializer())

    if "graph_dir" in self.setting:
      self.set_summary(self.sess, self.setting["graph_dir"])

    logger = None
    if "logger_params" in self.setting:
      logger = tf_logger.BaseLogger(**self.setting["logger_params"])
    self.logger = logger

    
  def set_saver(self, params):
    self.saver = tf.train.Saver(**params)

  def set_summary(self, sess, graphdir):
    self.summary = tf.summary.FileWriter(graphdir, sess.graph)

  def initialize_train(self, *args):
    self.all_metrics ={}

  def initialize_epoch(self, loader, epochs, batch_size, epoch):
    if self.logger is not None:
      self.logger.start_epoch(self, loader, epoch)    


  def finalize_epoch(self, loader, epochs, batch_size, epoch):    
    if self.logger is not None:
      self.logger.end_epoch(self, loader, epoch)    
    


  def end_train(self, loader):
    if self.logger is not None:
      self.logger.log_end(self, loader)

  def get_feed(self, itr):
    raise NotImplementedError()
    

  def train_batch(self, itr):
    fd = self.get_feed(itr)
    run_dict = {**self.trainers , **self.losses}
    train_batch = self.sess.run(run_dict, feed_dict=fd)    

    if self.logger is not None:
      self.logger.log_batch(train_batch)


  def train(self, loader, epochs, batch_size=32):
      self.initialize_train(loader, epochs, batch_size)
      for epoch in range(1, epochs + 1):
        n_iter, iter_ = loader["train"].get_data_iterators(batch_size, epoch=epoch)
        self.initialize_epoch(loader, epochs, batch_size, epoch)
        for itr in iter_:
          self.train_batch(itr)
        self.finalize_epoch(loader, epochs, batch_size, epoch)
      self.end_train(loader)


class SLBaseTrainer(TFBaseTrainer):
  def make_model(self, shape, n_classes, is_train=True):
    model_params = self.setting["model_arch"]
    model_params["shape"] = shape
    model_params["n_classes"] = n_classes
    return model_creator.make_model(model_params)

  def get_losses(self, inputs, models, loss_params):
    labels = inputs["target"]
    logits = models["logits"]
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
                                                          logits=logits))
    return {"loss":loss}
    
  def get_trainer(self, inputs, models, losses):
    trainer = {}
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(losses["loss"])
    trainer["train_op"] = train_op
    return trainer    

  def get_feed(self, itr):
    fd = {}
    fd[self.inputs["input"]] = itr[0]
    fd[self.inputs["is_train"]] = True
    fd[self.inputs["target"]] = itr[1]
    return fd
