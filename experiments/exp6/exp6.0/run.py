import os, sys
import toml
import argparse
from munch import Munch, munchify
from typing import Dict, List
import pandas as pd
import json
import torch
from torchvision import transforms


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp6.0")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results", "exp6.0")


sys.path.append(PROJ_DIR)
from utils import update_config, create_dir_for_file
from experiments.exp6.utils import train
import datasets


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.checkpoints_dir = CHECKPOINTS_DIR
    config.default.results_dir = RESULTS_DIR
    
    def _run_on_dataset(dataset:str):
        count_list = []
        for seed in config.sweep.seeds:
            count = train(config, dataset, seed, verbose=args.verbose)
            count_list.append(count)
            
        output_file = os.path.join(RESULTS_DIR, f"{dataset}.json")
        create_dir_for_file(output_file)
        with open(output_file, "w") as f:
            json.dump(count_list, f)
    
    if args.mnist:
        _run_on_dataset('MNIST')
    
    if args.cifar10:
        _run_on_dataset('CIFAR10')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
        help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
        help="run cifar10 experiment")
    parser.add_argument("--verbose", action="store_true", default=False,
        help="print progress information")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    config = toml.load(os.path.join(SCRIPTS_DIR,"exp6","exp6.0","config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)

    run(args, config)

