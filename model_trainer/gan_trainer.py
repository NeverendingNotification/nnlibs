#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 06:12:36 2018

@author: nn
"""

import numpy as np
import tensorflow as tf

from .tf_base_trainer import TFBaseTrainer
from . import model_creator
from . import tf_optimizer


class GanBase(TFBaseTrainer):
  def get_sample(self, size):
    rand = np.random.normal(size=[size, self.n_dim])
    return rand

  def generate_images(self, loader, image_num, metrics=[]):
    sample = self.get_sample(image_num)
    fd = {self.inputs["noise"]:sample,
          self.inputs["is_train"]:False
          }
#    imgs = self.sess.run(self.models["prediction"],
#                         feed_dict=fd)
    imgs = self.sess.run(self.models["generator"],
                         feed_dict=fd)
#    import numpy as np
#    imgs = np.concatenate([in_imgs, imgs], axis=2)
    imgs = loader.test.postprocess(imgs)
    return imgs
  

class GanBaseTrainer(GanBase):
  def make_model(self, loader, is_train=True):
    super().make_model(loader, is_train=is_train)
    model_params = self.setting["network_params"]
    model_params["shape"] = loader.get_shape()
    model_params["n_classes"] = loader.get_n_classes()
    self.n_dim = model_params["hidden_dim"]
    return model_creator.make_model(model_params, is_train=is_train)

  def get_losses(self, inputs, models, loss_params):
    with tf.name_scope("loss"):
      fake_logits = models["fake_dis"]
      real_logits = models["real_dis"]
      loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=fake_logits, labels=tf.zeros_like(fake_logits)))
      loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=real_logits, labels=tf.ones_like(real_logits)))
      loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=fake_logits, labels=tf.ones_like(fake_logits)))
    return {"loss_gen":loss_gen, "loss_dis_fake":loss_dis_fake, 
            "loss_dis_real":loss_dis_real}

    
  def get_trainer(self, inputs, models, losses):
    opt_params = self.setting["update_params"]
    g_opt, lrate = tf_optimizer.get_optimizer(self.epoch, opt_params)
    d_opt, lrate = tf_optimizer.get_optimizer(self.epoch, opt_params)
    self.lrate = lrate      
    trainer = {}
    g_train = g_opt.minimize(losses["loss_gen"],
                             var_list=models["gen_vars"])
    d_train = d_opt.minimize(losses["loss_dis_real"] + losses["loss_dis_fake"],
                             var_list=models["dis_vars"])
    trainer["train_op_d"] = d_train
    trainer["train_op_g"] = g_train
    trainer["step"] = tf.assign_add(self.global_step, 1)
    

#    with tf.control_dependencies([tf.assign_add(self.global_step, 1)]):
#      train_op = opt.minimize(losses["loss"])
#    trainer["train_op"] = train_op
    return trainer

#  def train(self, loader, epochs, batch_size=32):
#    raise NotImplementedError()

  def train_batch(self, itr, epoch):
#    fd = self.get_feed(itr)

    batch_size = len(itr[0])
    noise1 = self.get_sample(batch_size)
    fd_dis = {self.inputs["input"]:itr[0],
              self.inputs["is_train"]:True,
              self.inputs["noise"]:noise1}
    result_dis = self.sess.run({
        "train_dis":self.trainers["train_op_d"],
        "loss_dis_real":self.losses["loss_dis_real"],
        "loss_dis_fake":self.losses["loss_dis_fake"]},
        feed_dict=fd_dis)
              
    noise2 = self.get_sample(batch_size)
    fd_gen = {self.inputs["noise"]:noise2, self.inputs["is_train"]:True}
    result_gen = self.sess.run({
        "train_gen":self.trainers["train_op_g"],
        "loss_gen":self.losses["loss_gen"]},
        feed_dict=fd_gen)
    
    
    run_dict = {**result_dis , **result_gen}
    
#    train_batch = self.sess.run(run_dict, feed_dict=fd)
    if self.logger is not None:
      self.logger.log_batch(run_dict, loss_keys=["loss_dis_real",
                                                    "loss_dis_fake",
                                                    "loss_gen"])


        
class WGanBaseTrainer(GanBase):
  def make_model(self, loader, is_train=True):
    super().make_model(loader, is_train=is_train)
    model_params = self.setting["network_params"]
    model_params["shape"] = loader.get_shape()
    model_params["n_classes"] = loader.get_n_classes()
    self.n_dim = model_params["hidden_dim"]
    return model_creator.make_model(model_params, is_train=is_train)

  def get_losses(self, inputs, models, loss_params):
    with tf.name_scope("loss"):
      fake_logits = models["fake_dis"]
      real_logits = models["real_dis"]
      loss_dis = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
      loss_gen = tf.reduce_mean(- fake_logits)
#      loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#          logits=fake_logits, labels=tf.zeros_like(fake_logits)))
#      loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#          logits=real_logits, labels=tf.ones_like(real_logits)))
#      loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#          logits=fake_logits, labels=tf.ones_like(fake_logits)))
    return {"loss_gen":loss_gen, "loss_dis":loss_dis}

    
  def get_trainer(self, inputs, models, losses):
    opt_params = self.setting["update_params"]
    g_opt, lrate = tf_optimizer.get_optimizer(self.epoch, opt_params)
    d_opt, lrate = tf_optimizer.get_optimizer(self.epoch, opt_params)
    self.lrate = lrate      
    trainer = {}
    g_train = g_opt.minimize(losses["loss_gen"],
                             var_list=models["gen_vars"])
    d_train = d_opt.minimize(losses["loss_dis"],
                             var_list=models["dis_vars"])
    trainer["train_op_d"] = d_train
    trainer["train_op_g"] = g_train
    trainer["step"] = tf.assign_add(self.global_step, 1)
    with tf.name_scope("clipping"):
      clipping = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in models["dis_vars"]]
    trainer["clipping"] = clipping
    print(len(models["dis_vars"]))
    print(len(models["gen_vars"]))
    print(len(clipping))
#    with tf.control_dependencies([tf.assign_add(self.global_step, 1)]):
#      train_op = opt.minimize(losses["loss"])
#    trainer["train_op"] = train_op
    return trainer

#  def train(self, loader, epochs, batch_size=32):
#    raise NotImplementedError()

  def train_batch(self, itr, epoch):
#    fd = self.get_feed(itr)    
    iter_dis = 25 if epoch <=2 else 5
    batch_size = len(itr[0])
    for _ in range(iter_dis):
      noise1 = self.get_sample(batch_size)
      fd_dis = {self.inputs["input"]:itr[0],
                self.inputs["is_train"]:True,
                self.inputs["noise"]:noise1}
      result_dis = self.sess.run({
         "train_dis":self.trainers["train_op_d"],
         "loss_dis":self.losses["loss_dis"]},
         feed_dict=fd_dis)
      self.sess.run(self.trainers["clipping"])
              
    noise2 = self.get_sample(batch_size)
    fd_gen = {self.inputs["noise"]:noise2, self.inputs["is_train"]:True}
    result_gen = self.sess.run({
        "train_gen":self.trainers["train_op_g"],
        "loss_gen":self.losses["loss_gen"]},
        feed_dict=fd_gen)

    
    run_dict = {**result_dis , **result_gen}
    
#    train_batch = self.sess.run(run_dict, feed_dict=fd)
    if self.logger is not None:
      self.logger.log_batch(run_dict, loss_keys=["loss_dis",
                                                 "loss_gen"])

  
from .models import cnn
class WGanGpTrainer(GanBase):
  def make_model(self, loader, is_train=True):
    super().make_model(loader, is_train=is_train)
    model_params = self.setting["network_params"]
    model_params["shape"] = loader.get_shape()
    model_params["n_classes"] = loader.get_n_classes()
    self.n_dim = model_params["hidden_dim"]
    self.gp = model_params["gradient_penalty"]
    shape = model_params["shape"]
    with tf.name_scope("inputs"):
      in_imgs = tf.placeholder(tf.float32, shape=(None,) + shape)
      is_train = tf.placeholder(tf.bool)
      noise = tf.placeholder(tf.float32, shape=(None, self.n_dim))
    inputs = {"input":in_imgs, "is_train":is_train, "noise":noise}
    
    dis_name = "discriminator"
    gen_name = "generator"
    
    gen = cnn.get_generator(noise, shape, is_train,
                        model_params["gen_params"],
                        var_name=gen_name, act=tf.nn.tanh)
    fake_dis = cnn.get_discriminator(gen, is_train,
                                 model_params["discrim_params"],
                                 var_name=dis_name)
    real_dis = cnn.get_discriminator(in_imgs, is_train,
                                 model_params["discrim_params"],
                                 var_name=dis_name)

    # calculate gradient penalty
    with tf.name_scope("intermidate"):
      eps_shape = tf.shape(in_imgs)
      eps = tf.random_uniform(eps_shape[:1], minval=0, maxval=1.0)
      eps = tf.reshape(eps, [-1, 1, 1, 1])
      mid = in_imgs + eps * (gen - in_imgs)
      
      mid_dis = cnn.get_discriminator(mid, is_train,
                                   model_params["discrim_params"],
                                   var_name=dis_name)
    
      grads = tf.gradients(mid_dis, [mid])[0]

    
    tvars = tf.trainable_variables()
    gen_vars = [var for var in tvars if var.name.startswith(gen_name)]
    dis_vars = [var for var in tvars if var.name.startswith(dis_name)]
    
    models = {"fake_dis":fake_dis, "real_dis":real_dis,
              "mid_grads":grads, "generator":gen, 
              "gen_vars":gen_vars, "dis_vars":dis_vars}
    return inputs, models


#    return model_creator.make_model(model_params, is_train=is_train)

  def get_losses(self, inputs, models, loss_params):
    with tf.name_scope("loss"):
      fake_logits = models["fake_dis"]
      real_logits = models["real_dis"]
      loss_dis = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
      loss_gen = tf.reduce_mean(- fake_logits)
      
      
      slopes = tf.sqrt(1.0e-10 + tf.reduce_sum(tf.square(models["mid_grads"]), axis=[1, 2, 3]))
      gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    return {"loss_gen":loss_gen, "loss_dis":loss_dis, 
            "gradient_penalty":gradient_penalty}

    
  def get_trainer(self, inputs, models, losses):
    opt_params = self.setting["update_params"]
    g_opt, lrate = tf_optimizer.get_optimizer(self.epoch, opt_params)
    d_opt, lrate = tf_optimizer.get_optimizer(self.epoch, opt_params)
    self.lrate = lrate      
    trainer = {}
    g_train = g_opt.minimize(losses["loss_gen"],
                             var_list=models["gen_vars"])
    d_train = d_opt.minimize(losses["loss_dis"] + self.gp * losses["gradient_penalty"],
                             var_list=models["dis_vars"])
    trainer["train_op_d"] = d_train
    trainer["train_op_g"] = g_train
    trainer["step"] = tf.assign_add(self.global_step, 1)
#    with tf.control_dependencies([tf.assign_add(self.global_step, 1)]):
#      train_op = opt.minimize(losses["loss"])
#    trainer["train_op"] = train_op
    return trainer

#  def train(self, loader, epochs, batch_size=32):
#    raise NotImplementedError()

  def train_batch(self, itr, epoch):
#    fd = self.get_feed(itr)    
    iter_dis = 25 if epoch <=2 else 5
    batch_size = len(itr[0])
    for _ in range(iter_dis):
      noise1 = self.get_sample(batch_size)
      fd_dis = {self.inputs["input"]:itr[0],
                self.inputs["is_train"]:True,
                self.inputs["noise"]:noise1}
      result_dis = self.sess.run({
         "train_dis":self.trainers["train_op_d"],
         "loss_dis":self.losses["loss_dis"],
          "gradient_penalty": self.losses["gradient_penalty"]},
           feed_dict=fd_dis)
              
    noise2 = self.get_sample(batch_size)
    fd_gen = {self.inputs["noise"]:noise2, self.inputs["is_train"]:True}
    result_gen = self.sess.run({
        "train_gen":self.trainers["train_op_g"],
        "loss_gen":self.losses["loss_gen"]},
        feed_dict=fd_gen)

    
    run_dict = {**result_dis , **result_gen}
    
#    train_batch = self.sess.run(run_dict, feed_dict=fd)
    if self.logger is not None:
      self.logger.log_batch(run_dict, loss_keys=["loss_dis",
                                                 "loss_gen",
                                                 "gradient_penalty"])




