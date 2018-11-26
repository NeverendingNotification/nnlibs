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
  test_loader = loader.test
  if test_loader.get_type() == "feed":
    n_iter, iters = test_loader.get_data_iterators(batch_size,
                                                   is_train=False, random=False)
    preds = []
    cors = []
    for itr in tqdm(iters, total=n_iter):
      fd = {trainer.inputs["input"]:itr[0], trainer.inputs["is_train"]:False}
      predict = trainer.sess.run(pred, feed_dict=fd)
      preds.append(predict)
      cors.append(itr[1])
  else:
    trainer.sess.run(test_loader.iterator.initializer)
    tf_prediction = trainer.test_prediction if trainer.is_train \
                    else trainer.models["prediction"]
    tf_labels = loader.test.get_labels()
    preds = []
    cors = []
    try:
      while True:
        pred, cor = trainer.sess.run([tf_prediction,
                                      tf_labels])
        preds.append(pred)
        cors.append(cor)
    except tf.errors.OutOfRangeError:
      pass      
      
  prediction = np.concatenate(preds)
  y_pred = np.argmax(prediction, axis=1)
  correct = np.concatenate(cors)
  results = {}
  if "acc" in metrics:
    results["acc"] = np.mean(np.argmax(prediction, axis=1) == correct)
  if "ent" in metrics:
    epsilon =1e-10
    results["ent"] = -np.mean(np.sum(prediction * np.log(prediction + epsilon),
           axis=1))
  if "mse" in metrics:
    mse = np.mean(np.sqrt(np.power(prediction - correct, 2)))
    results["mse"] = mse    
  if "iou" in metrics:
    tp = np.sum((correct == 1) & (correct == y_pred))
    fn = np.sum((correct == 1) & (correct != y_pred))
    fp = np.sum((y_pred == 1) & (correct != y_pred))
    print(tp, fp, fn, len(prediction))
    results["iou"] = tp /(tp+fp+fn)
    
  return results

def get_metrics_generator(loader, trainer, batch_size=32, metrics=[]):
  images =  trainer.generate_images(loader, batch_size, metrics=metrics)
  return {}, images


