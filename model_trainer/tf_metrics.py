#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 05:38:32 2018

@author: nn
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

def get_metrics_classifier(loader, trainer, batch_size=32, metrics=[]):
  pred = trainer.models["prediction"]
  test_loader = loader["test"]
  n_iter, iters = test_loader.get_data_iterators(batch_size,
                                                 is_train=False, random=False)
  preds = []
  cors = []
  for itr in tqdm(iters, total=n_iter):
    fd = {trainer.inputs["input"]:itr[0], trainer.inputs["is_train"]:False}
    predict = trainer.sess.run(pred, feed_dict=fd)
    preds.append(predict)
    cors.append(itr[1])
  prediction = np.concatenate(preds)
  correct = np.concatenate(cors)
  
  results = {}
  if "acc" in metrics:
    results["acc"] = np.mean(np.argmax(prediction, axis=1) == correct)
  if "ent" in metrics:
    epsilon =1e-10
    results["ent"] = -np.mean(np.sum(prediction * np.log(prediction + epsilon),
           axis=1))
  return results



