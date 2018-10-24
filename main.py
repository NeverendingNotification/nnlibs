#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 07:33:54 2018

@author: nn
"""

import argparse
import os

from manager import get_loader, get_trainer, get_runner
import yaml_loader

def main(settings):  
  mode = settings["mode"]
  arc_type = settings["arc_type"]
  out_root = settings["out_root"] if "out_root" in settings else None
  
  loader_setting = settings["loader"]
  trainer_setting = settings["trainer"]
  runner_setting = settings["runner"]
  
  loader = get_loader(loader_setting, mode=mode, arc_type=arc_type)
  trainer = get_trainer(trainer_setting, loader, mode=mode, arc_type=arc_type,
                        out_root = out_root)
  runner = get_runner(runner_setting, loader, trainer, mode=mode, arc_type=arc_type)
  
  runner.run()
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--yml_file", type=str, default="test.yml")
  parser.add_argument("--test_dir", type=str, default=None)
  parser.add_argument("--out_dir", type=str, default=None)
  args = parser.parse_args()
  
  yml_file = args.yml_file
  test_dir = args.test_dir
  if test_dir is None:
    settings = yaml_loader.get_setting(yml_file)  
    main(settings)
  else:
    yml_files = [d for d in os.listdir(test_dir) if d.endswith(".yml")]
    for yml_file in yml_files:
      settings = yaml_loader.get_setting(os.path.join(test_dir, yml_file))
      settings["out_root"] = os.path.join(args.out_dir, settings["out_root"])
      print("start running :{}".format(yml_file))
      main(settings)
  