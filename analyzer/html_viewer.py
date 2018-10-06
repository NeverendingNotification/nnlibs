#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 11:34:27 2018

@author: nn
"""

import argparse
import os
import xml
import xml.etree.ElementTree as ET
import numpy as np

def make_html(img_paths, out_file):
  root = ET.Element("html")
  header = ET.SubElement(root, "header")
  body = ET.SubElement(root, "body")
  
  for img in img_paths:
    img_tag = ET.SubElement(body, "img", attrib={"src":img})
  ET.ElementTree(root).write(out_file)

def make_html_imgs(img_dir, out_file, sample_mode=100, random=True):
  files = np.array(os.listdir(img_dir))
  perm = np.random.permutation(len(files))
  targets = files[perm[:sample_mode]]
  img_paths = [os.path.join(img_dir, target) for target in targets]
  make_html(img_paths, out_file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("img_dir")
  parser.add_argument("out_file")
  args = parser.parse_args()
  make_html_imgs(args.img_dir, args.out_file)
