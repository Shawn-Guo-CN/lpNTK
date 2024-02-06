import os, sys
import toml
import argparse
from munch import Munch, munchify
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp7.1")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from utils import update_config, create_dir_for_file


def plot_in_axis(ax, df, size1:int, size2:int) -> None:
    ax.scatter(df[size1], 
               df[size2],
               color='blue',
               linewidths=0.2,
               alpha=0.1
              )
        
    corr = df[size1].corr(df[size2], method='pearson')
    ax.set_title(f"{str(size1)} vs {str(size2)} " + r'($\rho$=' + \
                 f'{corr.round(3)})', fontsize=16
                )
    
    ax.grid(True)
    plt.subplots_adjust(hspace=0.25)


def plot_result(dataset:str, num_points:int=1000, lr:float=0.1) -> None:
    result_path = os.path.join(RESULTS_DIR, 
                               'exp7.1', 
                               f"{dataset}_{lr}.csv"
                              )
    df = pd.read_csv(result_path)
    df = df.sample(n=num_points)
    
    sizes = list(df.columns[1:])
    size_combines = list(combinations(sizes, 2))
    
    figure, axes = plt.subplots(math.ceil(len(size_combines) / 3), 3, 
                                figsize=(10, 
                                         3 * math.ceil(len(size_combines) / 3)
                                        )
                               )
    
    for idx, (size1, size2) in enumerate(size_combines):
        plot_in_axis(axes[idx // 3, idx % 3], df, size1, size2)
    
    outpath = os.path.join(RESULTS_DIR,
                           'exp7.1',
                           f"{dataset}_{num_points}_{lr}.pdf"
                          )
    create_dir_for_file(outpath)
    plt.savefig(outpath, bbox_inches='tight', format='pdf', dpi=300)
    
    
def plot_single_row(dataset:str, num_points:int=1000, lr:float=0.1) -> None:
    """Plot the correlation between the largest size and all others in a row.
    """
    result_path = os.path.join(RESULTS_DIR, 
                               'exp7.1', 
                               f"{dataset}_{lr}.csv"
                              )
    df = pd.read_csv(result_path)
    df = df.sample(n=num_points)
    
    sizes = list(df.columns[1:])
    size_combines = [(sizes[0], size) for size in sizes[1:]]
    
    figure, axes = plt.subplots(1, len(size_combines), figsize=(35,5))
    
    for idx, (size1, size2) in enumerate(size_combines):
        plot_in_axis(axes[idx], df, size1, size2)
    
    outpath = os.path.join(RESULTS_DIR,
                           'exp7.1',
                           f"{dataset}_{num_points}_{lr}_row.pdf"
                          )
    create_dir_for_file(outpath)
    plt.savefig(outpath, bbox_inches='tight', format='pdf', dpi=300)


def print_corr(dataset:str, lr:float) -> None:
    result_path = os.path.join(RESULTS_DIR, 
                               'exp7.1', 
                               f"{dataset}_{lr}.csv"
                              )
    df = pd.read_csv(result_path)
    
    sizes = list(df.columns[1:])
    size_combines = list(combinations(sizes, 2))
    for size1, size2 in size_combines:
        print(f"Corr {size1} vs {size2}: \
                {df[size1].corr(df[size2], method='pearson')}")


def main(args:argparse) -> None:
    if args.mnist:
        print_corr("MNIST", args.lr)
        plot_result("MNIST", args.num_points, args.lr)
        plot_single_row("MNIST", args.num_points, args.lr)
        
    if args.cifar10:
        print_corr("CIFAR10", args.lr)
        plot_result("CIFAR10", args.num_points, args.lr)
        plot_single_row("CIFAR10", args.num_points, args.lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
                        help="Train on MNIST.")
    parser.add_argument("--cifar10", action="store_true", default=False, 
                        help="Train on CIFAR10.")
    parser.add_argument("--num_points", type=int, default=1000,
                        help="Number of points to plot for the dataset.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate.")
    parser.add_argument("--verbose", action="store_true", default=False, 
                        help="Print verbose output.")
    args, unknown = parser.parse_known_args()
    
    main(args)