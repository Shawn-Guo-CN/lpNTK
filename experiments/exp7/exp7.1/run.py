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
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp7.1")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from utils import update_config, create_dir_for_file, \
                    acquire_flock, release_flock, set_seed
from experiments.exp7.utils import construct_subset_indices, train
import datasets


def get_partition(config:Munch, dataset:str) -> Dict[int, List]:
    partition_file = os.path.join(config.default.results_dir, 
                                  'exp7.1', 
                                  f"{dataset}_partition.json"
                                 )
    
    size2subsets = None
    if os.path.exists(partition_file):
        size2subsets = json.load(open(partition_file, 'r'))
        size2subsets = {int(k): size2subsets[k] for k in size2subsets.keys()}
    else:
        size2subsets = construct_subset_indices(config, dataset)
        _lock_file_fd_ = acquire_flock(partition_file)
        json.dump(size2subsets, open(partition_file, 'w'))
        release_flock(_lock_file_fd_)
    
    return size2subsets


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.checkpoints_dir = CHECKPOINTS_DIR
    config.default.results_dir = RESULTS_DIR
    
    def _train_on_dataset_(dataset:str) -> None:
        size2subsets = get_partition(config, dataset)
        dataset_config = config[dataset]
        
        train_transform = eval(dataset_config.transform)
        train_dataset = eval('datasets.'+dataset)(root=DATA_DIR, 
                                                  train=True, 
                                                  download=True, 
                                                  transform=train_transform
                                                 )
        
        dataset_config = config[dataset]
        dataset_config.size = args.size
        config.default.update(dataset_config)
        config.default.update({'experiment':config.wandb.experiment})
    
        use_cuda = config.default.use_cuda and torch.cuda.is_available()
        set_seed(config.default.seed)
        device = torch.device("cuda" if use_cuda else "cpu")
    
        train_kwargs = {'batch_size':  config.default.batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True,
                           }
            train_kwargs.update(cuda_kwargs)
        
        all_ld = {}
        if args.verbose:
            print(f"Training on {dataset} with subset size {args.size}...")
        for subset in size2subsets[args.size]:
            ld = train(config, dataset, train_dataset, subset,
                       train_kwargs, device
                      )
            all_ld = {**all_ld, **ld}
        if args.verbose:
            print(f"Done training on {dataset} with subset size {args.size}.")
            
        csv_path = os.path.join(config.default.results_dir, 
                               'exp7.1', 
                               f"{dataset}_{config.default.lr}.csv"
                              )
        create_dir_for_file(csv_path)
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=['idx'])
            df['idx'] = size2subsets[config.default.sizes[0]][0]
        m = df['idx'].isin(all_ld.keys())
        df.loc[m, args.size] = df['idx'].map(all_ld)
        df.to_csv(csv_path, index=False)

    if args.mnist:
        _train_on_dataset_("MNIST")

    if args.cifar10:
        _train_on_dataset_("CIFAR10")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
        help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
        help="run cifar10 experiment")
    parser.add_argument("--size", type=int, default=4096,
        help="size of the subsets to train on")
    parser.add_argument("--verbose", action="store_true", default=False,
        help="print progress information")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    config = toml.load(os.path.join(SCRIPTS_DIR,"exp7","exp7.1","config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)

    run(args, config)