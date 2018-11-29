#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 05:38:32 2018

@author: nn
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

def get_iou(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_true == y_pred))
    fn = np.sum((y_true == 1) & (y_true != y_pred))
    fp = np.sum((y_pred == 1) & (y_true != y_pred))
    return tp / (tp + fn + fp)

def get_prediction(loader, trainer, batch_size=32):
  pred = trainer.models["prediction"]
  test_loader = loader.test
  if test_loader.get_type() == "feed":
    n_iter, iters = test_loader.get_data_iterators(batch_size,
                                                   is_train=False, random=False)
    preds = []
    for itr in tqdm(iters, total=n_iter):
      fd = {trainer.inputs["input"]:itr[0], trainer.inputs["is_train"]:False}
      predict = trainer.sess.run(pred, feed_dict=fd)
      preds.append(predict)
  else:
    trainer.sess.run(test_loader.iterator.initializer)
    tf_prediction = trainer.test_prediction if trainer.is_train \
                    else trainer.models["prediction"]

    preds = []
    try:
      while True:
        pred = trainer.sess.run(tf_prediction)
        preds.append(pred)
    except tf.errors.OutOfRangeError:
      pass      
  return np.concatenate(preds)


def predict_model(loader, trainer, batch_size=32):
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
  return preds, cors

def get_metrics_classifier(loader, trainer, batch_size=32, metrics=[],
                           threshold=0.5):
  preds, cors = predict_model(loader, trainer, batch_size=batch_size)
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
  if "auc" in metrics:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(correct, prediction[:, 1])
    results["auc"] = auc
  if "max_iou" in metrics:
    from sklearn.metrics import precision_recall_curve
    pre, rec, thr = precision_recall_curve(correct, prediction[:, 1])
    ious = pre * rec / (pre + rec - pre*rec + 1e-12)
    max_iou = np.max(ious)
    max_thr = thr[np.argmax(ious) - 1] 
    results["max_iou"] = max_iou
    results["max_iou_threshold"] = max_thr
  if "iou" in metrics:
#    tp = np.sum((correct == 1) & (correct == y_pred))
#    fn = np.sum((correct == 1) & (correct != y_pred))
#    fp = np.sum((y_pred == 1) & (correct != y_pred))
    y_pred1 = (prediction[:,  1] > threshold).astype(np.int)
#    y_pred2 = (prediction[:,  1] > 0.5).astype(np.int)
#    y_pred3 = (prediction[:,  1] > 0.75).astype(np.int)
    iou0 = get_iou(correct, y_pred1)      
    results["iou"] = iou0
    
  return results

def get_metrics_generator(loader, trainer, batch_size=32, metrics=[]):
  images =  trainer.generate_images(loader, batch_size, metrics=metrics)
  return {}, images


