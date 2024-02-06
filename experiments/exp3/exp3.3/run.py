import os, sys
import argparse
from collections import defaultdict, OrderedDict
from typing import List

from tqdm import tqdm, trange
import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets, transforms
import pickle


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp3.3")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from utils import create_dir_for_file


def sort_by_avg_sim(args: argparse, dataset:str) -> None:
    dataloader = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                    )
    
    if args.verbose:
        print(f"counting number of samples in class {args.class_id} ...")
    
    _range = trange(len(dataloader)) if args.verbose else range(len(dataloader))
    cls_idx_list = []
    idx2matidx = {}
    for i in _range:
        if dataloader[i][1] == args.class_id:
            idx2matidx[i] = len(cls_idx_list)
            cls_idx_list.append(i)
    
    if args.verbose:
        print(f"done, {len(cls_idx_list)} samples in class {args.class_id}.")
    if args.verbose:
        print("building similarity matrix...")
        
    sim_mat = np.zeros((len(cls_idx_list), len(cls_idx_list)), dtype=np.float32)
    _range = trange(len(cls_idx_list)) if args.verbose \
                else range(len(cls_idx_list))
    for _idx in _range:
        base_idx = cls_idx_list[_idx]
        assert dataloader[base_idx][1] == args.class_id
        
        npz_fname = os.path.join(DATA_DIR,
                                 dataset,
                                 f"{args.kernel}Kernel",
                                 "sorted",
                                 f"{base_idx}.npz"
                                )
        sorted_dis_vec = np.load(npz_fname)['data'].astype(np.float32)
        assert base_idx + sorted_dis_vec.shape[1] == len(dataloader)

        for i in range(sorted_dis_vec.shape[1]):
            if not dataloader[int(sorted_dis_vec[2, i])][1] == args.class_id:
                continue
            sim_mat[_idx][idx2matidx[int(sorted_dis_vec[2, i])]] = \
                sorted_dis_vec[0, i]
            sim_mat[idx2matidx[int(sorted_dis_vec[2, i])]][_idx] = \
                sorted_dis_vec[0, i]
                
    out_fpath = os.path.join(DATA_DIR, 
                             dataset,
                             f"{args.kernel}Kernel",
                             "sorted_by_avg_sim",
                             f"class_{args.class_id}_sim_mat.npz"
                            )
    create_dir_for_file(out_fpath)
    np.savez_compressed(out_fpath, data=sim_mat)
    
    if args.verbose:
        print("done.")
    if args.verbose:
        print("calculating average similarity...")
        
    _range = trange(len(cls_idx_list)) if args.verbose \
                else range(len(cls_idx_list))
    base_idx = np.asarray(cls_idx_list)
    avg_sim = np.mean(sim_mat, axis=1)
    sorted_by_avg_sim = np.stack([base_idx, avg_sim])
    sorted_by_avg_sim = \
        sorted_by_avg_sim[:, sorted_by_avg_sim[1].argsort()][::-1]
    
    if args.verbose:
        print("done.")
    
    out_fpath = os.path.join(DATA_DIR, 
                             dataset,
                             f"{args.kernel}Kernel",
                             "sorted_by_avg_sim",
                             f"class_{args.class_id}.npz"
                            )
    create_dir_for_file(out_fpath)
    np.savez_compressed(out_fpath, data=sorted_by_avg_sim)
    

def run(args: argparse) -> None:
    if args.mnist:
        sort_by_avg_sim(args, "MNIST")
    
    if args.cifar10:
        sort_by_avg_sim(args, "CIFAR10")

    if args.cifar100:
        sort_by_avg_sim(args, "CIFAR100")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--kernel", type=str, default="MaskSum",
        help="kernel to sort")
    parser.add_argument("--class-id", type=int, default=0,
        help="class index to preprocess")
    parser.add_argument("--verbose", action="store_true", default=False,
        help="verbose mode")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    run(args)