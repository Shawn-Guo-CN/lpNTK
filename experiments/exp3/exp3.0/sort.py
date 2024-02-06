import os, sys
import toml
import argparse
from munch import Munch, munchify

from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets, transforms


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


def sort_sample_kernel(base_idx:int, dataset:str, dataloader) -> None:
    npz_fname = os.path.join(DATA_DIR, 
                             dataset, 
                             f"{args.kernel}Kernel",
                             "raw",
                             f"{base_idx}.npz"
                            )
    dis_vec = np.load(npz_fname)['data'].astype(np.float32)

    assert base_idx + dis_vec.shape[0] == SAMPLE_SIZE[dataset]
    
    i_idx = base_idx * np.ones(dis_vec.shape[0])
    j_idx = base_idx + np.arange(dis_vec.shape[0])
    sorted_dis = np.stack([dis_vec, i_idx, j_idx])
    sorted_dis = sorted_dis[:, np.argsort(sorted_dis[0])[::-1]]

    out_path = os.path.join(DATA_DIR, 
                             dataset,
                             f"{args.kernel}Kernel",
                             "sorted",
                             f"{base_idx}.npz"
                            )
    create_dir_for_file(out_path)
    np.savez_compressed(out_path, data=sorted_dis)


def sort_kernel(args:argparse, dataset:str) -> None:
    dataloader = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )

    for base_idx in tqdm(range(args.start_idx, args.end_idx)):
        sort_sample_kernel(base_idx, dataset, dataloader)


def run(args:argparse) -> None:
    if args.mnist:
        sort_kernel(args, "MNIST")

    if args.cifar10:
        sort_kernel(args, "CIFAR10")

    if args.cifar100:
        sort_kernel(args, "CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--start-idx", type=int, default=0,
        help="start index of the GradMatrix calculation")
    parser.add_argument("--end-idx", type=int, default=None,
        help="end index of the GradMatrix calculation")
    parser.add_argument("--kernel", type=str, default="Sum",
        help="kernel to sort")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    run(args)