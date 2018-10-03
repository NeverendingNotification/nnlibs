#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:27:01 2018

@author: nn
"""

import yaml

def get_setting(in_file):
  with open(in_file, "r") as hndl:
    data = yaml.load(hndl)
  return data


if __name__ == "__main__":
  in_file = "test.yml"
  f = open(in_file, "r")
  data = yaml.load(f)
  print(data)
