from calendar import c
import os, sys
from venv import create
from sklearn.linear_model import ARDRegression
import toml
import argparse
from munch import Munch, munchify

import random
from collections import defaultdict
import torch
from tqdm import tqdm, trange
import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets, transforms
import pickle

from typing import List, Tuple


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp4")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")
NUM_CLS = {"MNIST": 10,
           "CIFAR10": 10,
           "CIFAR100": 100,
          }
SAMPLE_SHAPE = {"MNIST": (784,),
                "CIFAR10": (3072,),
                "CIFAR100": (3072,),
               }


sys.path.append(PROJ_DIR)
from utils import create_dir_for_file
from analysis.farthest_clustering import Cluster

def _get_cluster_list(args:argparse, dataset:str, cls_idx:int) -> List:
    pkl_fname = os.path.join(DATA_DIR,
                             dataset,
                             f"{args.kernel}Kernel",
                             f"far_cluster_{args.ratio}",
                             f"cluster_list_{cls_idx}.pkl",
                            )
    return Cluster.load(pkl_fname)

def prune_with_head(args: argparse, dataset:str, cls_idx:int) -> List:
    keep_sample_list = []
    cluster_list = _get_cluster_list(args, dataset, cls_idx)

    for cluster in cluster_list:
        keep_sample_list.append(cluster.head)
    
    return keep_sample_list

def prune_with_random(args: argparse, dataset:str, cls_idx:int) -> List:
    keep_sample_list = []
    cluster_list = _get_cluster_list(args, dataset, cls_idx)

    for cluster in cluster_list:
        random_idx = \
            random.sample(list(cluster.sample_od.keys()), 1)[0]
        keep_sample_list.append(random_idx)
    
    return keep_sample_list

def prune_with_node_size(args:argparse, dataset:str, cls_idx:int) -> List:
    cluster_list = _get_cluster_list(args, dataset, cls_idx)
    cluster_list = sorted(cluster_list, 
                          key=lambda x: len(x.sample_od), 
                          reverse=True
                         )
    
    keep_sample_list = []
    keep_sample_list += random.sample(list(cluster_list[0].sample_od.keys()),
                                      int(0.5*len(cluster_list[0].sample_od))
                                     )
    for cluster in cluster_list[1:]:
        keep_sample_list += list(cluster.sample_od.keys())

    return keep_sample_list

def prune_dataset(args:argparse, dataset:str) -> None:
    trainset = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )
    
    if args.verbose: print("Pruning dataset...")
    tmp_range = trange(NUM_CLS[dataset]) if args.verbose \
                    else range(NUM_CLS[dataset])
    keep_sample_idx_list = []
    for cls_idx in tmp_range:
        cls_keep_sample_list = []
        if args.method == "random":
            cls_keep_sample_list = prune_with_random(args, dataset, cls_idx)
        elif args.method == "head":
            cls_keep_sample_list = prune_with_head(args, dataset, cls_idx)
        elif args.method == "node":
            cls_keep_sample_list = prune_with_node_size(args, dataset, cls_idx)
        else:
            raise ValueError("Unknown method: {}".format(args.method))
        
        keep_sample_idx_list += cls_keep_sample_list
    if args.verbose: print("Done.")
    
    pruned_images = np.empty(SAMPLE_SHAPE[dataset])
    pruned_labels = []

    if args.verbose: print("Storing pruned dataset...")
    tmp_range = range(len(keep_sample_idx_list)) if not args.verbose else \
                    trange(len(keep_sample_idx_list))
    for idx in tmp_range:
        pruned_images = \
            np.vstack((pruned_images, 
                       trainset[keep_sample_idx_list[idx]][0].numpy().flatten()
                      ))
        pruned_labels.append(trainset.targets[keep_sample_idx_list[idx]])
    pruned_labels = np.array(pruned_labels)
    if args.verbose: print("Done.")

    args.ratio = round(pruned_images.shape[0] / len(trainset), 2)
    if args.verbose: print(f"Ratio: {args.ratio}")

    out_fname = os.path.join(
                    DATA_DIR,
                    dataset,
                    "pruned",
                    f"{args.kernel}_far_{args.method}_{args.ratio}.npy",
                )
    create_dir_for_file(out_fname)
    np.save(out_fname, (pruned_images, pruned_labels))


def run(args:argparse) -> None:
    if args.mnist:
        prune_dataset(args, "MNIST")

    if args.cifar10:
        prune_dataset(args, "CIFAR10")

    if args.cifar100:
        prune_dataset(args, "CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--kernel", type=str, default="Sum",
        help="kernel to sort")
    parser.add_argument("--ratio", type=float, default=0.1,
        help="ratio to keep")
    parser.add_argument("--method", type=str, default="head",
        help="prune method to use, can be head/random/node")
    parser.add_argument("--notes",   default=None)
    parser.add_argument("--verbose", action="store_true", default=False,
        help="print more info")
    args, unknown = parser.parse_known_args()

    run(args)