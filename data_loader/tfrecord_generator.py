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
import pandas as pd
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
      
      
def make_tf_record_files(files, labels, out_dir, offset=0, class_names=[],
                        param_filename="params.yml", split_size=20000,
                        random=True):
  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
  n_file = len(files)
  n_split = (n_file + split_size -1) //split_size
  files = np.array(files)
  labels = np.array(labels)
  if random:
    inds = np.random.permutation(n_file)
  else:
    inds = np.arange(n_file)
  classes = []
  filenames = []
  for n in range(n_split):
    n0 = n * split_size
    n1 = min((n+1) * split_size, n_file)
    ind = inds[n0:n1]
    t_files = files[ind]
    t_labels = labels[ind]
    print("step : ", n)
    out_file = os.path.join(out_dir, "record_{:02d}.tfrecord".format(n))
    with tf.python_io.TFRecordWriter(out_file) as writer:
      for f, file in enumerate(tqdm(t_files)):
        arr = cv2.imread(file)
        label = t_labels[f]
        filenames.append(os.path.basename(file))
        classes.append(label)
        example = tf.train.Example(features=tf.train.Features(feature={
                          "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[ind[f]])),
                          "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                          "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr.tostring()]))
                          }))
        writer.write(example.SerializeToString())
            
    
    
    
  num_classes = len(np.unique(labels))
  shape = arr.shape
  if len(class_names) == 0:
    class_names =[chr(ord("A") + i) for i in range(num_classes)]
  param_file = os.path.join(out_dir, param_filename)
  params ={
      "n_data":n_file,
      "num_classes":num_classes,
      "class_names":class_names,
      "shape":list(shape)
      }
  with open(param_file, "w") as hndl:
    yaml.dump(params, hndl)
  print(pd.Series(classes).value_counts())
  df = pd.DataFrame(classes, index=filenames, columns=["class"])
  df.to_csv(os.path.join(out_dir, "filenames.csv"))    

def get_train_test_indices(classes):
  inds = np.arange(len(classes))
  from sklearn.model_selection import StratifiedKFold
  spliter = StratifiedKFold(n_splits=5, shuffle=True)
  
  indices = []
  for ind, ind2 in spliter.split(inds, classes):
    indices.append((ind, ind2))
  return indices

      
def make_tf_records_from_directory(in_root, out_root, with_cv=False):
  dirs = os.listdir(in_root)
  all_files = []
  all_labels = []
  
  for d, dir_ in enumerate(dirs):
    files = os.listdir(os.path.join(in_root, dir_))
    all_files.extend([os.path.join(in_root, dir_, f) for f in files])
    all_labels.extend([d]*len(files))  

  all_files = np.array(all_files)
  all_labels = np.array(all_labels)
    
  if with_cv:
    indices = get_train_test_indices(all_labels)
    for i, (train_ind, test_ind)  in enumerate(indices):
      out_train = os.path.join(out_root, "cv_{:04d}".format(i), "Train")
      train_files = all_files[train_ind]
      train_labels = all_labels[train_ind]      
      out_test = os.path.join(out_root, "cv_{:04d}".format(i), "Test")
      test_files = all_files[test_ind]
      test_labels = all_labels[test_ind]      

      print("CV : ", i)
      print(pd.Series(train_labels).value_counts())
      print(pd.Series(test_labels).value_counts())
      make_tf_record_files(train_files, train_labels, out_train, class_names=dirs)
      make_tf_record_files(test_files, test_labels, out_test, class_names=dirs)

  else:
    make_tf_record_files(all_files, all_labels, out_root, class_names=dirs)

  
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
#  img_dir = "../../data/cifar10"
  in_root = "../../data/train"
  out_root = "../../data/train_cv"
  make_tf_records_from_directory(in_root, out_root, with_cv=True)
  
#  if not os.path.isdir(img_dir):
#    os.makedirs(img_dir)
#  tr, te = tf.keras.datasets.cifar10.load_data()
#  train = tr[0].reshape([-1, 32, 32, 3])
#  test = te[0].reshape([-1, 32, 32, 3])
#  make_tf_record_imgs(train, tr[1], os.path.join(img_dir, "Train"))
#  make_tf_record_imgs(test, te[1], os.path.join(img_dir, "Test"))
  

  