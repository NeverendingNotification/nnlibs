#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:24:58 2018

@author: nn
"""

import os

import numpy as np
import cv2


def get_data(dirname, resize=None):
  dirs = os.listdir(dirname)
  class_names = dirs
  classes = []
  images = []
  for c, class_ in enumerate(dirs):
    in_dir = os.path.join(dirname, class_)
    files = os.listdir(in_dir)
    for file in files:
      in_file = os.path.join(in_dir, file)
      arr = cv2.imread(in_file)
      if resize is not None:
        arr = cv2.resize(arr, tuple(resize))
      images.append(arr)
    classes.extend([c] * len(files))
  return np.array(images), np.array(classes), len(class_names)
        
def get_raw_loader(data_dir=None, train_name="Train", test_name="Test",
                   resize=None):
  assert data_dir is not None
  train_dir = os.path.join(data_dir, train_name)
  test_dir = os.path.join(data_dir, test_name)
  
  train_img, train_labels, n_classes = get_data(train_dir, resize=resize)
  print(train_img.shape, train_labels.shape)
  test_img, test_labels, n_classes = get_data(test_dir, resize=resize)
  print(test_img.shape, test_labels.shape)
  return (train_img, train_labels), (test_img, test_labels), n_classes

