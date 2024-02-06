from cmath import isnan
import os, sys
import argparse
from collections import defaultdict, OrderedDict
from tabnanny import check
from typing import List, Tuple

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


SAMPLE_SIZE = {
    'MNIST': 60000,
    'CIFAR10': 50000,
}


def is_sample_matrix_wrong(base_idx:int, 
                        dataset:str, 
                        verbose:bool=False,
                       ) -> bool:
    npz_fname = os.path.join(DATA_DIR, 
                             dataset, 
                             "GradMatrix",
                             f"{base_idx}.npz"
                            )
    grad_tensor = np.load(npz_fname)['data'].astype(np.float32)
    assert base_idx + grad_tensor.shape[0] == SAMPLE_SIZE[dataset]
    
    problem_idx_list = []
    _range = trange(grad_tensor.shape[0]) if verbose else \
                range(grad_tensor.shape[0])
    for shift_idx in _range:
        if np.isnan(grad_tensor[shift_idx]).any() or \
            np.isinf(grad_tensor[shift_idx]).any():
            return True
            
    return False


def gradmatrix_check(args: argparse, dataset:str) -> None:
    problem_idx_list = []
    
    _range = trange(SAMPLE_SIZE[dataset]) if args.verbose else \
                range(SAMPLE_SIZE[dataset])
    for base_idx in _range:
        if is_sample_matrix_wrong(base_idx, dataset, args.verbose):
            problem_idx_list.append(base_idx)
    
    out_fname = os.path.join(DATA_DIR,
                             dataset,
                             "problem_grad_idx.pkl"
                            )
    create_dir_for_file(out_fname)
    pickle.dump(problem_idx_list, open(out_fname, 'wb'))


def run(args: argparse) -> None:
    if args.mnist:
        gradmatrix_check(args, "MNIST")
    
    if args.cifar10:
        gradmatrix_check(args, "CIFAR10")

    if args.cifar100:
        gradmatrix_check(args, "CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--verbose", action="store_true", default=False,
        help="verbose mode")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    run(args)