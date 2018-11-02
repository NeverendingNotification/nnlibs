#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 05:24:53 2018

@author: nn
"""

import os
import yaml

import tensorflow as tf
import numpy as np
import cv2

from . import base_loader

#def init(self, num)

def get_decord_func(shape):

  def decord(example):
    features = tf.parse_single_example(
        example,
        features={
            "image":tf.FixedLenFeature([], tf.string),
            "label":tf.FixedLenFeature([], tf.int64),
            "index":tf.FixedLenFeature([], tf.int64)
            }
        )
  #  image = example.features.feature["image"].bytes_list.value[0]
  #  index = example.features.feature["index"].int64_list.value[0]
    image = features["image"]
    label = features["label"]
    index = features["index"]
      
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, shape)
    image = tf.cast(image ,tf.float32) / 255.0
    return image, label, index
  return decord

class TfRecordLoader(base_loader.BaseLoader):
  def __init__(self, record_dir, 
               batch_size,
               is_train=True,
               name="dataset",
               params_filename="params.yml"):
    super().__init__("tensor")
#record_file, n_data, shape, num_classes    
    record_files = [os.path.join(record_dir, f) \
                  for f in os.listdir(record_dir) \
                  if f.endswith(".tfrecord")]
    print("loading files :", record_files)
    param_file = os.path.join(record_dir, params_filename)
    with open(param_file, "r") as hndl:
      params = yaml.load(hndl)
    shape = tuple(params["shape"])
    
    with tf.name_scope(name):
      dataset = tf.data.TFRecordDataset(record_files)
      decord_func = get_decord_func(shape)
      dataset = dataset.map(decord_func)
#      dataset = dataset.map(augment)

      self.shape = shape
      self.n_classes = params["num_classes"]
      self.dataset = dataset
      self.n_data = params["n_data"]
      self.class_names = params["class_names"]
          
      if is_train:
        dataset = dataset.shuffle(batch_size * 50)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat(-1)
        iterator = dataset.make_one_shot_iterator()
      else:
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
      print("is_train", is_train)
      
    self.dataset = dataset
    self.iterator= iterator
    self.n_iter = (self.n_data + batch_size - 1) // batch_size
    self.batch = self.iterator.get_next()

  def get_batch(self):
    return self.batch
      
  def get_labels(self):
    return self.batch[1]
  
  def get_data_iterators(self, batch_size, epoch=0):
    return self.n_iter, range(self.n_iter)
    
#  dataset = dataset.shuffle(buffer_size=1028)

def check_record(tf_file):
  shape = (134, 183, 3)

  images = []
  c = 0
  for record in tf.python_io.tf_record_iterator(tf_file):
    example = tf.train.Example()
    example.ParseFromString(record)
    
    index = example.features.feature["index"].int64_list.value[0]
    image = example.features.feature["image"].bytes_list.value[0]
    
    image = np.fromstring(image, dtype=np.uint8)
    image = image.reshape(shape)
    images.append(image)
    c += 1
    print(c, index)
    if c == 3:
      break
  cv2.imwrite("tmp.png", np.concatenate(images))



def augment(image, label):
  """Placeholder for data augmentation."""
  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.
  image = tf.image.random_flip_left_right(image)
#  image = tf.image.random_brightness(image, 0.1)
#  image = tf.image.random_saturation(image, 0.5, 1.5)
  x = tf.random_uniform([1], minval=170, maxval=182, dtype=tf.int32)
  y = tf.random_uniform([1], minval=110, maxval=133, dtype=tf.int32)
  image = tf.random_crop(image, (y[0], x[0], 3))
#  rot = tf.random_uniform([1], minval=-1, maxval=1)
#  image = tf.contrib.image.rotate(image, np.pi * rot)
  image = tf.image.resize_images(image, (139, 189))
  return image, label


def batch_record(filename):
  dataset = tf.data.TFRecordDataset(filename)
  decord = get_decord_func((139, 189, 3))
  
  dataset = dataset.map(decord)
  dataset = dataset.map(augment)
  dataset = dataset.batch(8)
  dataset = dataset.shuffle(buffer_size=2)
#  iterator = dataset.make_one_shot_iterator()
  iterator = dataset.make_initializable_iterator()
  return iterator

def get_tfrecord_loader(input_dir="./", batch_size=32,
                        train_name="Train", test_name="Test"):
  train_file = os.path.join(input_dir, train_name)
  test_file = os.path.join(input_dir, test_name)
  
  train_loader = TfRecordLoader(train_file, batch_size, name="train_data")
  test_loader = TfRecordLoader(test_file, batch_size, is_train=False,
                               name="test_data")
  return train_loader, test_loader
    

if __name__ == "__main__":

  if False:
    tf_file = "/home/naoki/Document/ml_data/mtg/tfrecoords/10E.tfrecoord"
    
    tf_files = [
        "/home/naoki/Document/ml_data/mtg/tfrecoords/KLD.tfrecoord",
        "/home/naoki/Document/ml_data/mtg/tfrecoords/AER.tfrecoord"]
    iterator = batch_record(tf_files)
    img, index = iterator.get_next()
    
    with tf.Session() as sess:
      imgs = []
      for epoch in range(10):    
        indices = []
        sess.run(iterator.initializer)
        try:
          while True:
            im, i = sess.run([img, index])
            indices.append(i)
        except tf.errors.OutOfRangeError:
          print(epoch, indices[:4])
          imgs.append(np.concatenate(im))
  #      images.append(im)
  #      print(j, i)
      cv2.imwrite("tmp3.png", np.concatenate(imgs, axis=1))
  else:
    img_dir = "../../data/cifar10/Train"
    train_loader = TfRecordLoader(img_dir, 32)
    imgs, labels, indices = train_loader.get_batch()
    print(imgs.shape)
    print(labels)
    print(indices)
    with tf.Session() as sess:
      imgs = sess.run(imgs)
      height = int(np.sqrt(len(imgs)))
      width = len(imgs) // height
      img_hier = \
        np.concatenate(
          [np.vstack([imgs[j * height +i] for i in range(height)]) for j in range(width)
          ], axis=1)
      cv2.imwrite("tmp.png", img_hier)
