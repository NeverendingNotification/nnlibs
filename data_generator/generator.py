#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:13:45 2018

@author: nn
"""

import os

import numpy as np
import cv2

def make_data(path, class_, color, n_image, shape, ext="png"):
  colors ={ "r": (0, 0, 255), "g": (0, 255, 0), "b": (255, 0, 0)}
  
  for n in range(n_image):
    img_file = os.path.join(path, class_, "{:04d}.{}".format(n, ext))
    out = np.zeros(shape)
    noise = np.random.normal(loc=0, scale=30, size=shape)
    out[:] = colors[color] + noise    
    cv2.imwrite(img_file, out)
  

def generate_data(settings):
  mode = settings["mode"]
  cls_set = settings["class_setting"]
  img_shape = tuple(settings["img_size"]) + (3, )
  if mode == "class":
    root = settings["out_root"]
    for dir_ in settings["out_dirs"]:
      o_dir = os.path.join(root, dir_)
      for c, class_name in enumerate(cls_set["color_settings"]):
        cls_dir = os.path.join(o_dir, class_name)
        if not os.path.isdir(cls_dir):
          os.makedirs(cls_dir)
        make_data(o_dir, class_name, class_name, cls_set["img_counts"][c],
                  img_shape)
        
    pass
  else:
    raise NotImplementedError()
  
  
  