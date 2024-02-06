import os, sys
import toml
import argparse
from munch import Munch, munchify

from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets, transforms
from collections import defaultdict
import pickle
from tqdm import tqdm, trange


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp3")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from utils import create_dir_for_file


def build_selfmax_list(args:argparse, dataset:str) -> None:
    dataloader = eval('datasets.'+dataset)(DATA_DIR, 
                                            train=True, 
                                            download=True, 
                                            transform=transforms.ToTensor()
                                           )
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
        if sorted_kernel[2, 0] == sorted_kernel[1, 0]:
            cls2selfmax_list[dataloader[idx][1]].append(idx)

    cls2selfmax_fname = os.path.join(DATA_DIR,
                                     dataset,
                                     f"{args.kernel}Kernel",
                                     f"cls_to_selfmax.pkl",
                                    )
    create_dir_for_file(cls2selfmax_fname)
    with open(cls2selfmax_fname, 'wb') as f:
        pickle.dump(cls2selfmax_list, f)

    return cls2selfmax_list


def run(args:argparse) -> None:
    if args.mnist:
        build_selfmax_list(args, "MNIST")

    if args.cifar10:
        build_selfmax_list(args, "CIFAR10")

    if args.cifar100:
        build_selfmax_list(args, "CIFAR100")


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
    parser.add_argument("--verbose", action="store_true", default=False,
        help="verbose mode")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    run(args)