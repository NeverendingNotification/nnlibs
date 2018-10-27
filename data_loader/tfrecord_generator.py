#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:38:07 2018

@author: nn
"""

import os

import cv2
import yaml

import numpy as np
import tensorflow as tf
from tqdm import tqdm

  

def make_tf_recoord(in_dir, out_file):
  files = os.listdir(in_dir)
  with tf.python_io.TFRecordWriter(out_file) as writer:
    for f, file in enumerate(tqdm(files)):
      arr = cv2.imread(os.path.join(in_dir, file))
      example = tf.train.Example(features=tf.train.Features(feature={
                        "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[f])),
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr.tostring()]))
                        }))
      writer.write(example.SerializeToString())
      
      
def make_tf_record_imgs(imgs, labels, out_dir, offset=0, class_names=[],
                        param_filename="params.yml"):
  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
  out_file = os.path.join(out_dir, "record.tfrecord")
  with tf.python_io.TFRecordWriter(out_file) as writer:
    for f, img in enumerate(tqdm(imgs)):
      example = tf.train.Example(features=tf.train.Features(feature={
                        "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[f + offset])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[f]])),
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgs[f].tostring()]))
                        }))
      writer.write(example.SerializeToString())
  n_data = len(labels)
  num_classes = len(np.unique(labels))
  shape = imgs[0].shape
  if len(class_names) == 0:
    class_names =[chr(ord("A") + i) for i in range(num_classes)]
  param_file = os.path.join(out_dir, param_filename)
  params ={
      "n_data":n_data,
      "num_classes":num_classes,
      "class_names":class_names,
      "shape":list(shape)
      }
  with open(param_file, "w") as hndl:
    yaml.dump(params, hndl)
  
  

if __name__ == "__main__":
#  in_dir = "/home/naoki/Document/ml_data/mtg/cardlists/KLD"
#  out_file = "/home/naoki/Document/ml_data/mtg/tfrecoords/KLD.tfrecoord"
#  make_tf_recoord(in_dir, out_file)
  img_dir = "../../data/cifar10"
  if not os.path.isdir(img_dir):
    os.makedirs(img_dir)
  tr, te = tf.keras.datasets.cifar10.load_data()
  train = tr[0].reshape([-1, 32, 32, 3])
  test = te[0].reshape([-1, 32, 32, 3])
  make_tf_record_imgs(train, tr[1], os.path.join(img_dir, "Train"))
  make_tf_record_imgs(test, te[1], os.path.join(img_dir, "Test"))
  

  