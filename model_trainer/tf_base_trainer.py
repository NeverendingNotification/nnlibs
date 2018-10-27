#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:37:33 2018

@author: nn
"""

import os

from tqdm import tqdm
import tensorflow as tf

from .basic_trainer import BaseTrainer
from . import model_creator
from . import tf_logger
from .utils import tf_function
from . import tf_optimizer
from .models import cnn

class TFBaseTrainer(BaseTrainer):
  def __init__(self, trainer_setting):
    super().__init__(trainer_setting)

  
  def initialize_training(self, loader):
    self.sess = tf.Session()
    saver_setting = self.setting["saver_setting"] \
                    if "saver_setting" in self.setting \
                    else {}
    if "out_root" in self.setting:
      out_root = self.setting["out_root"]
    else:
      out_root = None
    self.set_saver(saver_setting)
    
 
    
    if "load_model_path" in self.setting:
      lmp =self.setting["load_model_path"]
      if out_root is not None:
        lmp = os.path.join(out_root, lmp)
      self.saver.restore(self.sess, lmp)
    else:
      self.sess.run(tf.global_variables_initializer())

    if "graph_dir" in self.setting:
      graph_dir = self.setting["graph_dir"]
      if out_root is not None:
        graph_dir = os.path.join(out_root, graph_dir)
      if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
      self.set_summary(self.sess, graph_dir)

    logger = None
    if "logger_params" in self.setting:
      logger_params = self.setting["logger_params"]
      if out_root is not None:
        logger_params["out_root"] = out_root
      logger = tf_logger.get_logger(self.setting["arc_type"],
                                    logger_params)
        
    self.logger = logger
    if "save_model_path" in self.setting:
      self.setting["save_model_path"] = os.path.join(logger.log_dir,
                  self.setting["save_model_path"])

    
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
      
    result = self.sess.run([self.epoch, self.epoch_op,
                            self.global_step, self.lrate])


  def end_train(self, loader):
    if self.logger is not None:
      self.logger.log_end(self, loader)
    if "save_model_path" in self.setting:
      self.saver.save(self.sess, self.setting["save_model_path"])
    tf.reset_default_graph()

  def get_feed(self, itr):
    raise NotImplementedError()
    

  def train_batch_feed(self, itr, epoch):
    fd = self.get_feed(itr)
    run_dict = {**self.trainers , **self.losses}
    train_batch = self.sess.run(run_dict, feed_dict=fd)    

    if self.logger is not None:
      self.logger.log_batch(train_batch)
      
  def train_batch(self, itr, epoch):
    run_dict = {**self.trainers , **self.losses}
    train_batch = self.sess.run(run_dict)    

    if self.logger is not None:
      self.logger.log_batch(train_batch)


  def train(self, loader, epochs, batch_size=32):
    batch_func = self.train_batch_feed if loader.get_type() == "feed" else \
                    self.train_batch
    
    self.initialize_train(loader, epochs, batch_size)
    for epoch in range(1, epochs + 1):
      n_iter, iter_ = loader.train.get_data_iterators(batch_size, epoch=epoch)
      self.initialize_epoch(loader, epochs, batch_size, epoch)
      for itr in tqdm(iter_, total=n_iter):
#        self.train_batch(itr, epoch)
        batch_func(itr, epoch)
      self.finalize_epoch(loader, epochs, batch_size, epoch)
    self.end_train(loader)

  def make_model(self, loader, is_train=True):
    with tf.name_scope("util"):
      self.epoch = tf.Variable(1, dtype=tf.int32, trainable=False, name="epoch")
      self.global_step = tf.Variable(0, name="global_step", trainable=False)
      self.epoch_op = tf.assign_add(self.epoch, 1)


class SLBaseTrainer(TFBaseTrainer):
  
  def make_inputs(self, loader, model_params, train=True):
    if loader.get_type() == "feed":
      shape = loader.get_shape()
      with tf.name_scope("inputs"):
        in_imgs = tf.placeholder(tf.float32, shape=(None,) + shape)
        is_train = tf.placeholder(tf.bool)
        trg_imgs = tf.placeholder(tf.int32, shape=(None,))
      inputs = {"input":in_imgs, "is_train":is_train, "target":trg_imgs}
    else:
      in_imgs, trg_imgs, _ = loader.get_batch()
      inputs = {"input":in_imgs, "is_train":train, "target":trg_imgs}
    return inputs
  
  def make_network(self, inputs, model_params):
    conv_last, feature, logits = cnn.get_neural_net(inputs["input"],
                                                    inputs["is_train"],
                                                    model_params["n_classes"],
                                                    model_params["conv_params"],
                                                    model_params["feature_type"],
                                                    model_params["mlp_params"]
                                                )
    
    pred = tf.nn.softmax(logits)
    models = {"conv_output":conv_last, "feature":feature,
              "logits":logits, "prediction":pred}
    return models
  
  def make_model(self, loader, is_train=True):
    super().make_model(loader, is_train=is_train)
    model_params = self.setting["network_params"]["classifier_params"]
    model_params["n_classes"] = loader.get_n_classes()
#    return model_creator.make_model(model_params, is_train=is_train)
#    inputs, models = cnn.get_network(model_params, is_train=is_train)
    inputs = self.make_inputs(loader.loader, model_params, train=is_train)
    models = self.make_network(inputs, model_params)
    if is_train and (loader.test is not None) and (loader.get_type() == "tensor"):
      test_inputs = self.make_inputs(loader.test, model_params, train=False)
      test_models = self.make_network(test_inputs, model_params)
      self.test_prediction = test_models["prediction"]
      self.test_labels = test_inputs["target"]
    return inputs, models

  def get_losses(self, inputs, models, loss_params):
    with tf.name_scope("loss"):
      labels = inputs["target"]
      logits = models["logits"]
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
                                                            logits=logits))
    return {"loss":loss}
    
  def get_trainer(self, inputs, models, losses):
    opt_params = self.setting["update_params"]
    opt, lrate = tf_optimizer.get_optimizer(self.epoch, opt_params)
    self.lrate = lrate      
    trainer = {}
    with tf.control_dependencies([tf.assign_add(self.global_step, 1)]):
      train_op = opt.minimize(losses["loss"])
    trainer["train_op"] = train_op
    return trainer    

  def get_evaluator(self, inputs, models):
    return {}


  def get_feed(self, itr):
    fd = {}
    fd[self.inputs["input"]] = itr[0]
    fd[self.inputs["is_train"]] = True
    fd[self.inputs["target"]] = itr[1]
    return fd

  def evaluate(self, loader, eval_params):
    from . import tf_metrics
    out = tf_metrics.get_metrics_classifier(loader, self, 
                                      metrics=["acc"])
    print(out)


class AEBaseTrainer(TFBaseTrainer):
  def make_model(self, loader, is_train=True):
    super().make_model(loader, is_train=is_train)
    model_params = self.setting["network_params"]
    model_params["shape"] = loader.get_shape()
    model_params["n_classes"] = loader.get_n_classes()
    return model_creator.make_model(model_params, is_train=is_train)

  def get_losses(self, inputs, models, loss_params):
    with tf.name_scope("loss"):
      targets = inputs["target"]
#      logits = models["logits"]
#      logits = models["prediction"]
#      self.n_pred = models["prediction"]
      self.n_pred = models["logits"]
      print(targets.shape)
      print(self.n_pred.shape)
      loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=targets, 
                                                          predictions=self.n_pred))
    return {"loss":loss}

  def get_trainer(self, inputs, models, losses):
    opt_params = self.setting["update_params"]
    opt, lrate = tf_optimizer.get_optimizer(self.epoch, opt_params)
    self.lrate = lrate      
    trainer = {}
    with tf.control_dependencies([tf.assign_add(self.global_step, 1)]):
      train_op = opt.minimize(losses["loss"])
    trainer["train_op"] = train_op
    return trainer

  def get_evaluator(self, inputs, models):
    return {}

  def get_feed(self, itr):
    fd = {}
    fd[self.inputs["input"]] = itr[0]
    fd[self.inputs["is_train"]] = True
    fd[self.inputs["target"]] = itr[0]
    return fd

  def evaluate(self, loader, eval_params):
    from . import tf_output
    tf_output.output(loader, self, eval_params)
    
  def generate_images(self, loader, image_num, metrics=[]):
    in_imgs = loader.test.get_random_images(image_num)
    fd = {self.inputs["input"]:in_imgs,
          self.inputs["is_train"]:False
          }
#    imgs = self.sess.run(self.models["prediction"],
#                         feed_dict=fd)
    imgs = self.sess.run(self.n_pred,
                         feed_dict=fd)
    import numpy as np
    imgs = np.concatenate([in_imgs, imgs], axis=2)
    imgs = loader.test.postprocess(imgs)
    return imgs
        
