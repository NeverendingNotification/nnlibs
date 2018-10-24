#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 06:28:12 2018

@author: nn
"""
import os

import numpy as np
import tensorflow as tf
import cv2


def output(loader, trainer, params):
  out_dir = params["eval_dir"]
  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
  n_out = params["n_out"]
  inputs = trainer.inputs
  logits = trainer.models["logits"]
  test_loader = loader.test
  n_data = test_loader.n_data
  perm = np.random.permutation(n_data)[:n_out]
  imgs = test_loader.get_images(perm)
  feed = {inputs["input"]:imgs, inputs["is_train"]:False}
  out = trainer.sess.run(logits, feed_dict=feed)
  for p in range(len(perm)):
    o_img = test_loader.postprocess(imgs[p])
    g_img = test_loader.postprocess(out[p])
    
    cv2.imwrite(os.path.join(out_dir, "{:04d}.png".format(p)), 
                             np.concatenate([o_img, g_img], axis=1))
  

  