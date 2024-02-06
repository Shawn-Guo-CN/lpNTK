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
from analysis.hierarchical_clustering import UnionTracker, Cluster


def prune_with_uniform_dist(args:argparse, 
                            dataloader:datasets.VisionDataset,
                            cls_idx:int,
                           ) -> List:
    keep_sample_list = []
    tmp_range = trange(len(dataloader)) if args.verbose else \
                    range(len(dataloader))
    for idx in tmp_range:
        if dataloader[idx][1] == cls_idx:
            keep_sample_list.append(idx)
    keep_sample_list = random.sample(keep_sample_list, 
                                     int(args.ratio * len(keep_sample_list))
                                    )

    return keep_sample_list


def prune_with_self_not_max(args:argparse,
                            dataloader:datasets.VisionDataset,
                            cls_idx:int,
                            self_is_max_idx_list:List,
                           ) -> List:
    keep_sample_list = []
    tmp_range = trange(len(dataloader)) if args.verbose else \
                    range(len(dataloader))
    for idx in tmp_range:
        if dataloader[idx][1] == cls_idx and idx in self_is_max_idx_list:
            keep_sample_list.append(idx)

    keep_sample_list = random.sample(keep_sample_list, 
                                     int(args.ratio * len(keep_sample_list))
                                    )
    
    return keep_sample_list


def recontruct_hier_cluster_tree(args:argparse, 
                                 dataset:str, 
                                 cls_idx:int
                                ) -> Tuple[List, UnionTracker, List]:
    cls_idx_list_fname = os.path.join(DATA_DIR,
                                      dataset,
                                      f"{args.kernel}_hier_cluster",
                                      f"cls_idx_{cls_idx}.pkl",
                                     )
    union_tracker_fname = os.path.join(DATA_DIR,
                                       dataset,
                                       f"{args.kernel}_hier_cluster",
                                       f"union_tracker_cls{cls_idx}.npy",
                                        )
    with open(cls_idx_list_fname, 'rb') as f:
        cls_idx_list = pickle.load(f)
    
    union_tracker = UnionTracker()
    union_tracker.load(union_tracker_fname)

    cluster_list = []
    sampleidx2clusteridx = {}
    for idx, sample_idx in enumerate(cls_idx_list):
        cluster_list.append(
            Cluster(**{'type': 'leaf', 'sample_idx': sample_idx})
        )
        sampleidx2clusteridx[sample_idx] = idx

    for idx in range(union_tracker.linkage_matrix.shape[0]):
        src_cluster_idx = int(union_tracker.linkage_matrix[idx, 0])
        tgt_cluster_idx = int(union_tracker.linkage_matrix[idx, 1])
        norm = union_tracker.linkage_matrix[idx, 2]
        assert \
            idx + len(cls_idx_list) == int(union_tracker.linkage_matrix[idx, 3])
        cluster_list.append(Cluster(**{
            'type': 'internal',
            'lchild': src_cluster_idx,
            'rchild': tgt_cluster_idx,
            'norm': norm,
            'lchild_sample_idx': cluster_list[src_cluster_idx].sample_idx_list,
            'rchild_sample_idx': cluster_list[tgt_cluster_idx].sample_idx_list
        }))

    return cluster_list, union_tracker, cls_idx_list


def prune_with_hier_tree_class(args:argparse, 
                                dataset:str,
                                dataloader:datasets.VisionDataset,
                                self_is_max_idx_list:List,
                                cls_idx:int,
                               ) -> List:
    keep_sample_idx_list = []

    cluster_list, union_tracker, sample_idx_list = \
        recontruct_hier_cluster_tree(args, dataset, cls_idx)

    norm_threshold = \
        union_tracker.linkage_matrix[-int(len(sample_idx_list) * args.ratio), 2]

    queue = [len(cluster_list) - 1]
    while queue:
        cluster_idx = queue.pop(0)
        if cluster_list[cluster_idx].type == 'leaf':
            keep_sample_idx_list.append(
                    cluster_list[cluster_idx].sample_idx_list[0]
                )
        elif cluster_list[cluster_idx].norm < norm_threshold:
            queue.append(cluster_list[cluster_idx].lchild)
            queue.append(cluster_list[cluster_idx].rchild)
        elif cluster_list[cluster_idx].norm > norm_threshold:
            keep_sample_idx_list.append(
                random.sample(cluster_list[cluster_idx].sample_idx_list, 1)[0]
            )

    return keep_sample_idx_list


def get_self_is_max_idx_list(args:argparse, 
                             dataset:str,
                             dataloader:datasets.VisionDataset,
                            ) -> List:
    cls2selfmax_fname = os.path.join(DATA_DIR,
                                     dataset,
                                     f"{args.kernel}Kernel",
                                     f"cls_to_selfmax.pkl",
                                    )

    if os.path.exists(cls2selfmax_fname):
        return pickle.load(open(cls2selfmax_fname, 'rb'))

    cls2selfmax_list = defaultdict(list)
    tmp_range = trange(len(dataloader)) if args.verbose else \
                    range(len(dataloader))
    for idx in tmp_range:
        sorted_kernel_fname = os.path.join(DATA_DIR,
                                           dataset,
                                           f"{args.kernel}Kernel",
                                           "sorted",
                                           f"{idx}.npz"
                                           )
        sorted_kernel = np.load(sorted_kernel_fname)['data'].astype(np.float32)
        if sorted_kernel[1, 0] == sorted_kernel[2, 0]:
            cls2selfmax_list[dataloader[idx][1]].append(idx)

    with open(cls2selfmax_fname, 'wb') as f:
        pickle.dump(cls2selfmax_list, f)

    return cls2selfmax_list


def prune_dataset(args:argparse, dataset:str) -> None:
    trainset = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )
    
    cls2selfmax_list = []
    if not args.method == "uniform":
        if args.verbose: print("Building self-is-max list...")
        cls2selfmax_list = get_self_is_max_idx_list(
            args, dataset, trainset
        )
        if args.verbose: print("Done.")

    if args.verbose: print("Pruning dataset...")
    keep_sample_idx_list = []
    for cls_idx in trange(NUM_CLS[dataset]):
        cls_keep_sample_list = []
        if args.method == "uniform":
            cls_keep_sample_list = prune_with_uniform_dist(
                                       args, trainset, cls_idx
                                   )
        elif args.method == "selfmax":
            cls_keep_sample_list = prune_with_self_not_max(
                                       args, trainset, cls_idx, 
                                       cls2selfmax_list[cls_idx]
                                   )
        elif args.method == "hier":
            cls_keep_sample_list = prune_with_hier_tree_class(
                                        args, 
                                        dataset, 
                                        trainset,
                                        cls2selfmax_list[cls_idx],
                                        cls_idx
                                   )
        
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
                    f"{args.kernel}_hier_{args.method}_{args.ratio}.npy",
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
    parser.add_argument("--method", type=str, default="hier",
        help="prune method to use, can be hier/uniform/selfmax")
    parser.add_argument("--notes",   default=None)
    parser.add_argument("--verbose", action="store_true", default=False,
        help="print more info")
    args, unknown = parser.parse_known_args()

    run(args)