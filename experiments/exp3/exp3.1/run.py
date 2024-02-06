import os, sys
from sklearn.linear_model import ARDRegression
import toml
import argparse
from munch import Munch, munchify

from tqdm import tqdm, trange
import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets, transforms
import pickle


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp3")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")

sys.path.append(PROJ_DIR)
from utils import create_dir_for_file
from analysis.hierarchical_clustering import UnionTracker, Cluster


def hierarchical_cluster_class(args:argparse, dataset:str) -> None:
    trainset = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )
    npz_name = os.path.join(DATA_DIR,
                            dataset,
                            f"{args.kernel}Kernel",
                            "class_merged",
                            f"{args.class_id}.npz"
                           )
    sorted_norm = np.load(npz_name)['data'].astype(np.float32)

    union_tracker = UnionTracker()
    cluster_list = []
    cls_idx_list = []
    idx2cluster_dict = {}

    for i, (_, label) in enumerate(trainset):
        if label == args.class_id:
            cls_idx_list.append(i)
            cluster_list.append(Cluster(**{'type': 'leaf', 'sample_idx': i}))
            idx2cluster_dict[i] = cluster_list[-1].idx

    j = 0
    for i in trange(sorted_norm.shape[1]):
        cur_norm = sorted_norm[0, i]
        src_idx = int(sorted_norm[1, i])
        tgt_idx = int(sorted_norm[2, i])

        if not tgt_idx in cls_idx_list:
            continue

        src_cluster_idx = idx2cluster_dict[src_idx]
        tgt_cluster_idx = idx2cluster_dict[tgt_idx]

        if src_cluster_idx == tgt_cluster_idx:
            continue

        l_sample_list = cluster_list[src_cluster_idx].sample_idx_list
        r_sample_list = cluster_list[tgt_cluster_idx].sample_idx_list
        if set(l_sample_list).issubset(set(r_sample_list)) \
            or set(r_sample_list).issubset(set(l_sample_list)):
            continue

        assert trainset[tgt_idx][1] == args.class_id

        cluster_list.append(Cluster(**{'type': 'internal',
                                       'lchild': src_cluster_idx,
                                       'rchild': tgt_cluster_idx,
                                       'norm': cur_norm,
                                       'lchild_sample_idx': l_sample_list,
                                       'rchild_sample_idx': r_sample_list
        }))
        new_cluster_idx = cluster_list[-1].idx
        assert new_cluster_idx == Cluster.count - 1

        for sample_idx in cluster_list[-1].sample_idx_list:
            idx2cluster_dict[sample_idx] = new_cluster_idx

        idx2cluster_dict[src_idx] = new_cluster_idx
        idx2cluster_dict[tgt_idx] = new_cluster_idx
        union_tracker.add(src_cluster_idx, tgt_cluster_idx, cur_norm, new_cluster_idx)

        j += 1
        if j >= len(cls_idx_list)-1:
            break
        del l_sample_list
        del r_sample_list

    union_tracker_fname = os.path.join(DATA_DIR,
                                       dataset,
                                       f"{args.kernel}Kernel",
                                       "hier_cluster",
                                       f"union_tracker_cls{args.class_id}.npy"
                                      )
    create_dir_for_file(union_tracker_fname)
    union_tracker.save(union_tracker_fname)

    cls_idx_list_fname = os.path.join(DATA_DIR,
                                      dataset,
                                      f"{args.kernel}Kernel",
                                      "hier_cluster",
                                      f"cls_idx_{args.class_id}.pkl"
                                      )
    create_dir_for_file(cls_idx_list_fname)
    with open(cls_idx_list_fname, 'wb') as f:
        pickle.dump(cls_idx_list, f)


def run(args:argparse) -> None:
    if args.mnist:
        hierarchical_cluster_class(args, "MNIST")

    if args.cifar10:
        hierarchical_cluster_class(args, "CIFAR10")

    if args.cifar100:
        hierarchical_cluster_class(args, "CIFAR100")


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
    parser.add_argument("--class-id", type=int, default=0,
        help="class index to preprocess")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    run(args)