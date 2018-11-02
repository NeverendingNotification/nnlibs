#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 05:24:53 2018

@author: nn
"""

import tensorflow as tf
import numpy as np
import cv2

#def init(self, num)


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

def decord(example):
  features = tf.parse_single_example(
      example,
      features={
          "image":tf.FixedLenFeature([], tf.string),
          "index":tf.FixedLenFeature([], tf.int64)
          }
      )
#  image = example.features.feature["image"].bytes_list.value[0]
#  index = example.features.feature["index"].int64_list.value[0]
  image = features["image"]
  index = features["index"]
    
  image = tf.decode_raw(image, tf.uint8)
  image = tf.reshape(image, (139, 189, 3))
  return image, index


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
  dataset = dataset.map(decord)
  dataset = dataset.map(augment)
#  dataset = dataset.shuffle(buffer_size=1028)
  dataset = dataset.batch(8)
  iterator = dataset.make_one_shot_iterator()
  iterator = dataset.make_initializable_iterator()
  return iterator


if __name__ == "__main__":
  tf_file = "/home/naoki/Document/ml_data/mtg/tfrecoords/10E.tfrecoord"
  
  tf_files = [
      "/home/naoki/Document/ml_data/mtg/tfrecoords/KLD.tfrecoord",
      "/home/naoki/Document/ml_data/mtg/tfrecoords/AER.tfrecoord"]
#  cnt = len(list(tf.python_io.tf_record_iterator(tf_file)))
#  example = next(tf.python_io.tf_record_iterator(tf_file))
#  key = tf.train.Example.FromString(example)
#  print(key)
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
        print(epoch, indices[0], indices[-1])
        imgs.append(np.concatenate(im))
#      images.append(im)
#      print(j, i)
    cv2.imwrite("tmp3.png", np.concatenate(imgs, axis=1))
    """
    im, i = sess.run([img, index])
    cv2.imwrite("tmp.png", np.concatenate(im))
    """