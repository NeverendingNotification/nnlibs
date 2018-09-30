#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 08:03:51 2018

@author: nn
"""

from data_loader import basic_loader

def get_loader(loader_params):
  loader = basic_loader.get_data_loader(loader_params)
  return loader
  


if __name__ == "__main__":
  loader_params = {}
  loader_params["data_type"] = "mnist"
  loader = get_loader(loader_params)
  print(loader)
  train_loader = loader["train"]
  arr = train_loader.get_images([0], raw=True)  
  import numpy as np
  print(arr.shape, np.min(arr), np.max(arr))
  out = arr[0].astype(np.uint8)
  import cv2
  cv2.imshow("test", out)
  cv2.waitKey(3000)
  