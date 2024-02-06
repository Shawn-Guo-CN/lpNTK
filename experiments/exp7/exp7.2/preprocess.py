import os, sys
import toml
import argparse
from munch import Munch, munchify
from typing import Dict, List
import pandas as pd
import json
import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict
from tqdm import trange


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp7.2")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from utils import update_config, rebuild_datasets_with_index_list
from experiments.exp7.utils import construct_subset_indices, train


def build_control_data4cls(dataset:str, num:int, cls_idx:int) -> None:
    npz_name = os.path.join(DATA_DIR,
                            dataset,
                            f"MaskSumKernel",
                            "class_merged",
                            f"{cls_idx}.npz"
                           )
    sorted_norm = np.load(npz_name)['data'].astype(np.float32)
    
    target_idx = int(sorted_norm[1, 0])
    
    target_idx_sorted_norm = []
    for i in range(sorted_norm.shape[1]):
        if sorted_norm[1, i] == target_idx or sorted_norm[2, i] == target_idx:
            target_idx_sorted_norm.append(sorted_norm[:, i])
    
    tgt_in_top = False
    for i in range(num):
        if target_idx_sorted_norm[i][1] == target_idx and \
            target_idx_sorted_norm[i][2] == target_idx:
            tgt_in_top = True

    top_num_list = target_idx_sorted_norm[1:num+1] if tgt_in_top else \
                       target_idx_sorted_norm[:num]
    mid_num_list = target_idx_sorted_norm[ \
                       (len(target_idx_sorted_norm) - num) // 2: \
                       (len(target_idx_sorted_norm) + num) // 2
                   ]
    bottom_num_list = target_idx_sorted_norm[-num:]
    
    if top_num_list[-1][0] < 10 * bottom_num_list[0][0]:
        print(f'class: {cls_idx}, gap is \
                {top_num_list[-1][0]}/{bottom_num_list[0][0]}')
    
    def _extract_idx(tgt:int, sort_list:List) -> List[int]:
        idx_list = []
        for i in range(len(sort_list)):
            if int(sort_list[i][1]) == tgt:
                idx_list.append(int(sort_list[i][2]))
            elif int(sort_list[i][2]) == tgt:
                idx_list.append(int(sort_list[i][1]))
            else:
                raise ValueError("Either index matches with target.")
        return idx_list
    
    easy_idx_list = \
        [target_idx] + _extract_idx(target_idx, top_num_list)
    mid_idx_list = \
        [target_idx] + _extract_idx(target_idx, mid_num_list)
    hard_idx_list = \
        [target_idx] + _extract_idx(target_idx, bottom_num_list)
    assert len(easy_idx_list)==len(mid_idx_list)==len(hard_idx_list)==num+1
    
    return easy_idx_list, mid_idx_list, hard_idx_list, [target_idx]


def build_control_datasets(dataset:str, config:Munch) -> None:
    """
    Build the datasets with controlled learning difficulty.
    
    This function is to build the dataset with more I/U/C samples for a specific
    sample in the given dataset. The samples are chosen based on the statistics
    from the function `get_sim_statistics`.
    
    NOTE: in the current version, I simply choose the sample that has the max 
    average similarity to the samples in the same class. This is not a good
    heuristic, but it's a good start. In this way, we can reuse the results from
    EXP4.3.
    
    Args:
        dataset: The dataset we are going to build the controlled datasets for.
        config: The configuration used for the experiment.
        
    Returns:
        None
        
    Outputs:
        Dataset files.
    """
    easy_idx_list = []
    mid_idx_list = []
    hard_idx_list = []
    original_idx_list = []
    if config.verbose: print('building control datasets...')
    tmp_range = trange(config.num_classes) if config.verbose else \
                    range(config.num_classes)
    for i in tmp_range:
        _easy_idx_list, _hard_idx_list, _mid_idx_list, _original_idx_list = \
            build_control_data4cls(dataset, config.num_samples, i)
        easy_idx_list += _easy_idx_list
        mid_idx_list  += _mid_idx_list
        hard_idx_list += _hard_idx_list
        original_idx_list += _original_idx_list
    if config.verbose: print('done')
    
    easy_out_path = os.path.join(DATA_DIR, 
                                 dataset, 
                                 "easy",
                                 f'{config.num_samples}.npy'
                                )
    rebuild_datasets_with_index_list(dataset, 
                                     easy_idx_list, 
                                     easy_out_path,
                                     config.verbose
                                    )
    
    mid_out_path = os.path.join(DATA_DIR, 
                                 dataset, 
                                 "mid",
                                 f'{config.num_samples}.npy'
                                )
    rebuild_datasets_with_index_list(dataset, 
                                     mid_idx_list, 
                                     mid_out_path,
                                     config.verbose
                                    )
    
    hard_out_path = os.path.join(DATA_DIR, 
                                 dataset, 
                                 "hard",
                                 f'{config.num_samples}.npy'
                                )
    rebuild_datasets_with_index_list(dataset, 
                                     hard_idx_list, 
                                     hard_out_path,
                                     config.verbose
                                    )
    
    original_out_path = os.path.join(DATA_DIR, 
                                 dataset, 
                                 "original",
                                 f'{config.num_samples}.npy'
                                )
    rebuild_datasets_with_index_list(dataset, 
                                     original_idx_list, 
                                     original_out_path,
                                     config.verbose
                                    )


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.checkpoints_dir = CHECKPOINTS_DIR
    config.default.results_dir = RESULTS_DIR
    
    if args.mnist:
        config.default.update(config['MNIST'])
        config = config.default
        config.verbose = args.verbose
        build_control_datasets('MNIST', config)
    
    if args.cifar10:
        config.default.update(config['CIFAR10'])
        config = config.default
        config.verbose = args.verbose
        build_control_datasets('CIFAR10', config)


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
    
    config = toml.load(os.path.join(SCRIPTS_DIR,"exp7","exp7.2","config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)
    
    run(args, config)