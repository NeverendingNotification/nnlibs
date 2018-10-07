#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 07:33:54 2018

@author: nn
"""

from manager import get_loader, get_trainer, get_runner
import yaml_loader

def main(settings):  
  mode = settings["mode"]
  loader_setting = settings["loader"]
  trainer_setting = settings["trainer"]
  runner_setting = settings["runner"]
  
  loader = get_loader(loader_setting, mode=mode)
  trainer = get_trainer(trainer_setting, loader, mode=mode)
  runner = get_runner(runner_setting, loader, trainer, mode=mode)
  
  runner.run()
  

if __name__ == "__main__":
  settings = yaml_loader.get_setting("test.yml")  
  main(settings)
  