from calendar import c
import os, sys
from collections import defaultdict, OrderedDict
import argparse
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


def prune_with_class_merged(
    args:argparse,
    dataset:str, 
    cls_idx:int, 
    head:bool
) -> List:
    npz_name = os.path.join(DATA_DIR,
                            dataset,
                            f"{args.kernel}Kernel",
                            "class_merged",
                            f"{cls_idx}.npz"
                           )
    sorted_norm = np.load(npz_name)['data'].astype(np.float32)
    
    idx2sortnorm = defaultdict(OrderedDict)
    for i in range(sorted_norm.shape[1]):
        src_idx = int(sorted_norm[1, i])
        tgt_idx = int(sorted_norm[2, i])
        idx2sortnorm[src_idx][tgt_idx] = sorted_norm[0, i]
        idx2sortnorm[tgt_idx][src_idx] = sorted_norm[0, i]

    list_idx_avg = []
    for idx in list(idx2sortnorm.keys()):
        avg_norm = np.mean(list(idx2sortnorm[idx].values()))
        list_idx_avg.append((idx, avg_norm))
    list_idx_avg = sorted(list_idx_avg, key=lambda x: x[1], reverse=head)

    sample_num = int(2 * (1. - args.ratio) * len(list_idx_avg))
    keep_sample_idx_list = \
        random.sample(list_idx_avg[:sample_num], k=int(sample_num * 0.5)) + \
        list_idx_avg[sample_num:]
    keep_sample_idx_list = [x[0] for x in keep_sample_idx_list]
    
    return keep_sample_idx_list


def prune_with_sorted_class_norm(
    args:argparse,
    dataset:str,
    cls_idx:int,
    head:bool=True
) -> List:
    npz_name = os.path.join(DATA_DIR,
                            dataset,
                            f"{args.kernel}Kernel",
                            "sorted_by_avg_sim",
                            f"class_{cls_idx}.npz"
                           )
    sorted_sim = np.load(npz_name)['data'].astype(np.float32)
    list_idx_avg = [int(x) for x in sorted_sim[1, :].tolist()]
    
    sample_num = int((1. - args.ratio) * len(list_idx_avg))
    keep_sample_idx_list = list_idx_avg[sample_num:] if head else \
        list_idx_avg[:-sample_num]

    return keep_sample_idx_list


def prune_dataset(args:argparse, dataset:str) -> None:
    trainset = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )
    keep_sample_idx_list = []
    
    prune_func = prune_with_class_merged if args.class_merged \
                    else prune_with_sorted_class_norm
    
    if args.verbose: print("Pruning dataset...")
    tmp_range = trange(NUM_CLS[dataset]) if args.verbose \
                    else range(NUM_CLS[dataset])
    
    for cls_idx in tmp_range:
        cls_keep_sample_list = []
        if args.method == "head":
            cls_keep_sample_list = prune_func(args, dataset, cls_idx, True)
        elif args.method == "tail":
            cls_keep_sample_list = prune_func(args, dataset, cls_idx, False)
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
                    f"{args.kernel}_avg_{args.ratio}.npy",
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
    parser.add_argument("--class-merged", action="store_true", default=False,
        help="use class merged files in data/dataset/kernel/class_merged")
    parser.add_argument("--kernel", type=str, default="MaskSum",
        help="kernel to sort")
    parser.add_argument("--ratio", type=float, default=0.9,
        help="ratio to keep")
    parser.add_argument("--method", type=str, default="head",
        help="prune method to use, can be head/tail")
    parser.add_argument("--notes",   default=None)
    parser.add_argument("--verbose", action="store_true", default=False,
        help="print more info")
    args, unknown = parser.parse_known_args()

    run(args)