#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 08:03:51 2018

@author: nn
"""

import numpy as np
from . import preset_loader
  
  


class LabelLoader:
  def __init__(self, data, labels, n_classes, preprocess=None, postprocess=None):
    self.preprocess = preprocess
    self.postprocess = postprocess
    if preprocess is not None:
      data = preprocess(data)
    self.data = data
    self.labels = labels
    self.n_classes = n_classes
    assert len(self.data) == len(self.labels)
    self.n_data = len(self.data)
    self.indices = np.arange(self.n_data)
    self.shape = self.data[0].shape

    
  
  def get_n_iter(self, batch_size):
    n_iter = (self.n_data + batch_size - 1) // batch_size
    return n_iter

  """
  This function should be over
  """
  def get_output(self, data, labels, indices, is_train):
    return data, labels, indices

  def get_images(self, indices, raw=False):
    if raw and self.postprocess is not None:
      return self.postprocess(self.data[indices])
    else:    
      return self.data[indices]
  
  def get_iter(self, batch_size, is_train, random):
    if random:
      perm = np.random.permutation(self.n_data)
    else:
      perm = np.arange(self.n_data)
    n_iter = self.get_n_iter(batch_size)

    for n in range(n_iter):
      i0 = n * batch_size
      i1 = min( (n + 1) * batch_size, self.n_data)
      inds = self.indices[perm[i0:i1]]
      data = self.get_images(inds)
      labels = self.labels[inds]
      indices = self.indices[inds]
      yield self.get_output(data, labels, indices, is_train)
      
  
  def get_data_iterators(self, batch_size, is_train=True, random=True, epoch=0):
    n_iter = self.get_n_iter(batch_size)
    iter_ = self.get_iter(batch_size, is_train, random)
    return n_iter, iter_

def preprocess_0_1(data):
  return (data/255.0)
def postprocess_0_1(data):
  return (data * 255)

def preprocess_1_1(data):
  return ((data -127.5)/255.0)
def postprocess_1_1(data):
  return ((data) + 1.0) * 255.0 / 2.0


def get_data_loader(loader_params):
  data_type = loader_params["data_type"]
  loader = {}
  pre = preprocess_0_1
  post = postprocess_0_1
  if data_type == "raw":
    pass
  else:
    train, test, n_classes = preset_loader.load_data(data_type)
    loader["train"] = LabelLoader(train[0], train[1], n_classes,
                                  preprocess=pre, postprocess=post
                                  )
    loader["test"] = LabelLoader(test[0], test[1], n_classes,
                                  preprocess=pre, postprocess=post
                                  )
    
    
  return loader


