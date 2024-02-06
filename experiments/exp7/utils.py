#!/usr/bin/python
# Author:  S Guo (s.guo@ed.ac.uk)
# Purpose: To provide some utility functions for the experiment 7, including
#          1. train a model on a given dataset and return learning difficulty
#          2. 
# Created: 2022-12-08


import os
from munch import Munch
import wandb
import numpy as np
from typing import List, Dict
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


from utils import set_seed, create_dir_for_file, create_dir, get_l2distance
from models import LeNet, ResNet18, ResNet50
import datasets


def construct_subset_indices(config:Munch, 
                             dataset:str
) -> Dict[int, List[List[int]]]:
    """Construct indices of samples in subsets for the following training.
       
    For a give dataset, we construct a dictionary where the key is the class
    and the value is a quad-tree stored in a list. Suppose the size of the
    subsets are [4096, 1024, 256, 64, 16, 4, 1], then root of the tree for a
    give class is then a list containing 4096 indices of samples in that
    class, and the root is stored in the first element of the list. The next 
    4 nodes in the list are the children of the root, and each of them is a
    list containing 1024 (mutually exclusive) indices of samples in that 
    class. So on and so forth.
    
    Then, these indices for each class will be merged into a dict where the 
    keys are the classes and the values are lists containing lists of equal 
    size. For example, if the sizes are [4096, 1024, 256, 64, 16, 4, 1], then
    the keys are the 4096, 1024, 256, 64, 16, 4, 1, (X) and the values are 
    lists containing $ 4096 / X $ lists of size X from all classes.
       
    Args:
        config: Munch object containing the configuration of the experiment.
        dataset: str, name of the benchmark dataset, e.g. MNIST.
    
    Returns:
        size2subsets: dict, key is the size and value is a list of lists.
    """
    
    args = config.default
    dataset_config = config[dataset]
    args.update(dataset_config)
    args.update({'experiment':config.wandb.experiment})
    
    train_dataset = eval('datasets.'+dataset)(root=args.data_dir, 
                                              train=True, 
                                              download=True
                                             )
    cls2idx_list = {}
    for _, cls, idx in train_dataset:
        if cls not in cls2idx_list.keys(): cls2idx_list[cls] = []
        cls2idx_list[cls].append(idx)
    cls_size = [len(cls2idx_list[cls]) for cls in cls2idx_list.keys()]
    
    for i in range(len(args.sizes) - 1):
        assert args.sizes[i] / args.sizes[i+1] == 4 and args.sizes[-1] == 1, \
            "The sizes of subsets should be the powers of 4."
    
    cls2subsets = {}
    for cls in cls2idx_list.keys():
        subset_list = []
        idx_uni_set = random.sample(cls2idx_list[cls], k=args.sizes[0])
        random.shuffle(idx_uni_set)
        
        for idx in idx_uni_set:
            subset_list.append([idx])
            
        _cur = 0
        while len(subset_list[_cur]) < args.sizes[0]:
            assert len(subset_list[_cur]) == len(subset_list[_cur+1]) and \
                len(subset_list[_cur]) == len(subset_list[_cur+2]) and \
                    len(subset_list[_cur]) == len(subset_list[_cur+3]), \
                    "The sizes of consecutive 4 elements should be equal."
            new_list = subset_list[_cur] + subset_list[_cur+1] + \
                          subset_list[_cur+2] + subset_list[_cur+3]
            subset_list.append(new_list)
            _cur += 4
        
        subset_list.reverse()
        cls2subsets[cls] = subset_list
    
    _cur = 0
    size2subsets = {}
    for s in args.sizes:
        size2subsets[s] = []
        num_subsets = args.sizes[0] // s
        for i in range(num_subsets):
            _subsets_sum = []
            for cls in range(args.num_classes):
                _subsets_sum += cls2subsets[cls][_cur]
            size2subsets[s].append(_subsets_sum)
            _cur += 1
    
    return size2subsets


def train(config:Munch, 
          dataset:str,
          train_dataset:torch.utils.data.Dataset, 
          subset_idx:List,
          train_kwargs:Dict,
          device:torch.device, 
         ) -> Dict[int, float]:
    """Train a model on a given subset, and return the learning difficulty.
    
    For a given benchmark dataset (e.g. MNIST), we train a model on a given
    list of indices of samples. During the training, we define the learning
    difficulty as the accumulated L2 distance between the predicted logits 
    and ground-truth one-hot labels. At the endo of the training, we return
    the learning difficulty along with sample indices.
    
    Args:
        config: Munch object containing the configuration of the experiment.
        dataset: str, name of the benchmark dataset, e.g. 'MNIST'.
        train_dataset: torch.utils.data.Dataset, the whole training dataset.
        subset_idx: list, indices of samples in the subset.
        train_kwargs: dict, keyword arguments for the training dataloader.
        device: torch.device, device to run the model.
    
    Return:
        ld: dict, key is the sample index and value is the learning difficulty.
    """
    
    subtrain_dataset = torch.utils.data.Subset(train_dataset, subset_idx)
    train_loader = torch.utils.data.DataLoader(subtrain_dataset, **train_kwargs)
    
    model = eval(config.default.model)(num_classes=config.default.num_classes
                                       ).to(device)
    optimiser = eval('optim.' + config.default.optim)(
                        model.parameters(), 
                        lr=config.default.lr, 
                        momentum=config.default.momentum, 
                        weight_decay=config.default.weight_decay
                    )
    if config.default.model == 'LeNet':
        scheduler = StepLR(optimiser, step_size=2, gamma=config.default.gamma)
    else:
        scheduler = CosineAnnealingLR(optimiser, T_max=200)

    ld = defaultdict(lambda: 0.0)
    
    for _ in range(1, config.default.epochs + 1):
        for data, target, dataidx in train_loader:
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            l2 = get_l2distance(output, target)
            for i in range(len(dataidx)):
                ld[dataidx.numpy()[i]] += l2[i]
            loss.backward()
            optimiser.step()
           
        scheduler.step()
    return ld

if __name__ == '__main__':
    train()