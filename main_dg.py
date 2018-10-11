#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 07:33:54 2018

@author: nn
"""


import yaml_loader
from data_generator import generator

def main(settings):  
  generator.generate_data(settings)

if __name__ == "__main__":
  settings = yaml_loader.get_setting("gen.yml")  
  main(settings)
  