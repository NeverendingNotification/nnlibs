#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 06:38:37 2018

@author: nn
"""

class BaseLoader:
  def __init__(self, type_):
    self.loader_type = type_
    
  def get_type(self):
    return self.loader_type
