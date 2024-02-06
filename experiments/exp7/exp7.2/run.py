import os, sys
import toml
import argparse
from munch import Munch, munchify
from typing import Dict, List
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
plt.style.use('bmh')
import itertools
import math


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp7.2")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from utils import update_config, update_munch_config, \
                    get_l2distance, create_dir_for_file
from pruned_datasets import PrunedMNIST, PrunedCIFAR10
from models import LeNet, ResNet18, ResNet50


def get_dataset_fname(dataset:str, mode:str, num_samples:int) -> str:
    return os.path.join(DATA_DIR, dataset, mode, f"{num_samples}.npy")


def train_on_dataset(config:Munch, dataset:str, mode:str) -> Dict[int, float]:
    args = config.default
    dataset_config = config[dataset]
    args.update(dataset_config)
    args.update({'experiment':config.wandb.experiment})
    
    use_cuda = args.use_cuda and torch.cuda.is_available()
    # set_seed(args.seed) # cannot fix seed for multiple runs
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False,
                       }
        train_kwargs.update(cuda_kwargs)
    
    train_transform = eval(config[dataset].transform)
    train_set = eval(f'Pruned{dataset}')(
                        get_dataset_fname(dataset, mode, args.num_samples),
                        transform = train_transform
                    )
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    
    
    model = eval(config[dataset].model)(num_classes=args.num_classes).to(device)
    optimiser = eval('optim.' + args.optim)(model.parameters(), 
                                            lr=args.lr, 
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay
                                           )
    
    ld_list = []
    for _ in range(args.epochs):
        ld_epoch = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            l2 = get_l2distance(output, target)
            ld_epoch.append(l2)
            loss.backward()
            optimiser.step()
        ld_list.append(np.concatenate(ld_epoch))
            
    return np.asarray(ld_list)


def get_original_idx(dataset:str, mode:str, num_samples:int) -> int:
    original_fname = get_dataset_fname(dataset, 'original', num_samples)
    mode_fname = get_dataset_fname(dataset, mode, num_samples)
    
    original_samples, _ = np.load(original_fname, allow_pickle=True)
    original_samples = original_samples[1:]
    mode_samples, _ = np.load(mode_fname, allow_pickle=True)
    mode_samples = mode_samples[1:]
    
    mode_idx_list = []
    for i in range(original_samples.shape[0]):
        orignal_sample = original_samples[i]
        for idx in range(mode_samples.shape[0]):
            mode_sample = mode_samples[idx]
            if np.all(orignal_sample == mode_sample):
                mode_idx_list.append(idx)
                break
    
    return mode_idx_list


def plot_ld_curves(config:Munch, dataset:str, mode2ld_tensor:Dict) -> None:
    assert mode2ld_tensor['easy'].shape[0] == config.num_train and \
             mode2ld_tensor['easy'].shape[1] == config.epochs and\
               mode2ld_tensor['easy'].shape[2] == config.num_classes
    
    mode2label = {
        # 'original': 'Stand-alone',
        'easy': 'More interchangable',
        'mid': 'Medium interchangable',
        'hard': 'More non-interchangeable'
    }
    mode2color = {
        # 'original': 'blue',
        'easy': 'red',
        'mid': 'yellow',
        'hard': 'purple'
    }
    
    figure, axes = plt.subplots(math.ceil(config.num_classes / 2),
                                2,
                                figsize=(20, 
                                         6 * math.ceil(
                                                config.num_classes\
                                               / 2
                                             )
                                        )
                               )
    
    def _plot_in_axis(ax, cls_idx:int) -> None:
        for mode in mode2label.keys():
            ld_mat = mode2ld_tensor[mode][:, :, cls_idx]
            ld_mat = ld_mat[~np.isnan(ld_mat).any(axis=1) ,:]
            
            _label = mode2label[mode]
            
            mean = np.mean(ld_mat, axis=0)
            up = mean + np.std(ld_mat, axis=0)
            down = mean - np.std(ld_mat, axis=0)

            ax.plot(mean, 
                    label=_label, 
                    color=mode2color[mode]
                   )
            ax.fill_between(range(len(mean)), 
                            up, 
                            down,
                            color=mode2color[mode],
                            alpha=0.2
                           )
            
            ax.set_xlabel('Training epochs')
            ax.set_ylabel(r'L2 distance between $y$ and $\hat{y}$')
            ax.set_title(f"Class {cls_idx}")
            if cls_idx == config.num_classes - 1:
                ax.legend()
            
    for cls_idx in range(config.num_classes):
        _plot_in_axis(axes[cls_idx // 2, cls_idx % 2], cls_idx)
    plt.subplots_adjust(hspace=0.2)
    
    out_fname = os.path.join(RESULTS_DIR, 
                             'exp7.2', 
                             f"{dataset}_ld_control_{config.lr}lr" + \
                                f"_{config.num_samples}samples.pdf"
                            )
    create_dir_for_file(out_fname)
    plt.savefig(out_fname, format='pdf', bbox_inches='tight')
    
    
def plot_ld_all_together(
    config:Munch, dataset:str, mode2ld_tensor:Dict
    ) -> None:
    assert mode2ld_tensor['easy'].shape[0] == config.num_train and \
             mode2ld_tensor['easy'].shape[1] == config.epochs and\
               mode2ld_tensor['easy'].shape[2] == config.num_classes
    
    mode2label = {
        # 'original': 'Stand-alone',
        'easy': 'More interchangable',
        'mid': 'Medium interchangable',
        'hard': 'More non-interchangeable'
    }
    mode2color = {
        # 'original': 'blue',
        'easy': 'red',
        'mid': 'yellow',
        'hard': 'purple'
    }

    figure, ax = plt.subplots(1, 1, figsize=(15, 6))
    
    for mode in mode2label.keys():
        ld_mat = mode2ld_tensor[mode][:, :, :].mean(axis=-1)
        ld_mat = ld_mat[~np.isnan(ld_mat).any(axis=1) ,:]
        print(f"for {mode}, the shape of ld_mat is: {ld_mat.shape}")
            
        _label = mode2label[mode]
            
        mean = np.mean(ld_mat, axis=0)
        up = mean + np.std(ld_mat, axis=0)
        down = mean - np.std(ld_mat, axis=0)

        ax.plot(mean, 
                label=_label, 
                color=mode2color[mode]
               )
        ax.fill_between(range(len(mean)), 
                        up, 
                        down,
                        color=mode2color[mode],
                        alpha=0.2
                       )
            
        ax.set_xlabel('Training epochs')
        ax.set_ylabel(r'L2 distance between $y$ and $\hat{y}$')
        ax.legend()

    out_fname = os.path.join(RESULTS_DIR, 
                             'exp7.2', 
                             f"{dataset}_ld_control_{config.lr}lr" + \
                                f"_{config.num_samples}samples_all.pdf"
                            )
    create_dir_for_file(out_fname)
    plt.savefig(out_fname, format='pdf', bbox_inches='tight')


def train_on_benchmark(config:Munch, dataset:str) -> None:
    config.default = update_munch_config(config.default, config[dataset])
    config = config.default
    
    ld_out_path = os.path.join(
                      RESULTS_DIR, 
                      'exp7.2', 
                      f"{dataset}_ld_control_{config.lr}lr" + \
                                f"_{config.num_samples}samples.npy",
                  )
    
    if os.path.isfile(ld_out_path):
        print(f"File {ld_out_path} already exists. Skipping...")
        mode2ld_tensor = np.load(ld_out_path, allow_pickle=True).item()
    else:
        mode2original_idx_dict = {}
        for mode in ['original', 'easy', 'mid', 'hard']:
            mode2original_idx_dict[mode] = \
                get_original_idx(dataset, mode, config.num_samples)
    
        mode2ld_tensor = {}
        for mode in ['original', 'easy', 'mid', 'hard']:
            _ld_tensor = []
            for _ in range(config.num_train):
                _ld_mat = train_on_dataset(config, dataset, mode)
                _ld_mat = _ld_mat[:, mode2original_idx_dict[mode]]
                _ld_tensor.append(_ld_mat)
            mode2ld_tensor[mode] = np.asarray(_ld_tensor)
        
        create_dir_for_file(ld_out_path)
        np.save(ld_out_path, mode2ld_tensor)
    
    plot_ld_curves(config, dataset, mode2ld_tensor)
    plot_ld_all_together(config, dataset, mode2ld_tensor)


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.results_dir = RESULTS_DIR
    
    if args.mnist:
        train_on_benchmark(config, "MNIST")
    
    if args.cifar10:
        train_on_benchmark(config, "CIFAR10")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="verbose mode")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    config = toml.load(os.path.join(SCRIPTS_DIR,"exp7","exp7.2","config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)

    run(args, config)