#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 07:33:54 2018

@author: nn
"""

import copy
import argparse
import os

from manager import get_loader, get_trainer, get_runner
import yaml_loader

def main(settings):  
  mode = settings["mode"]
  with_cv = "with_cv" in settings
  if not with_cv:
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
  else:
    arc_type = settings["arc_type"]
    out_root = settings["out_root"] if "out_root" in settings else None
    trainer_setting = settings["trainer"]
    runner_setting = settings["runner"]
        
    loader_setting = settings["loader"]
    in_root = loader_setting["raw_dir_params"]["input_dir"]
    cv_dirs = (os.listdir(in_root))
    print(cv_dirs)

    evals = []    
    for c, cv_dir in enumerate(cv_dirs[:]):
      tr_set = copy.deepcopy(trainer_setting)
      lo_set = copy.deepcopy(loader_setting)
      ru_set = copy.deepcopy(runner_setting)
      out_dir = os.path.join(out_root, "cv_{}".format(c))
      if "cv_fix_dir" in loader_setting:
        in_dir = loader_setting["cv_fix_dir"]
      else:
        in_dir = os.path.join(in_root, cv_dir)
        
      lo_set["raw_dir_params"]["input_dir"] = in_dir
      loader = get_loader(lo_set, mode=mode, arc_type=arc_type)
      trainer = get_trainer(tr_set, loader, mode=mode, arc_type=arc_type,
                            out_root = out_dir)
      runner = get_runner(ru_set, loader, trainer, mode=mode, arc_type=arc_type)
      
      
      out = runner.run()
      if mode == "eval":
        evals.append(out["iou"])
    if mode == "eval":
      print(evals)
      import numpy as np
      print(np.mean(evals))
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--yml_file", type=str, default="test.yml")
  parser.add_argument("--test_dir", type=str, default=None)
  parser.add_argument("--out_dir", type=str, default=None)
  parser.add_argument("--replace_out", action="store_true")
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
      if args.replace_out:
        settings["out_root"] = os.path.join(args.out_dir, settings["out_root"])
      print("start running :{}".format(yml_file))
      main(settings)
  