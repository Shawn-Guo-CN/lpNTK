import os
from random import shuffle
from munch import Munch
import wandb
from collections import defaultdict
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np

from utils import set_seed, create_dir_for_file, create_dir, get_l2distance
from models import LeNet, ResNet18, ResNet50
import datasets


def train(config:Munch, dataset:str, seed:int=None, verbose:bool=False):
    args = config.default
    dataset_config = config[dataset]
    args.update(dataset_config)
    args.update({'experiment':config.wandb.experiment})
    if seed is not None: args.seed = seed
    
    use_cuda = args.use_cuda and torch.cuda.is_available()
    set_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_transform = eval(dataset_config.transform)
    dataset1 = eval('datasets.'+dataset)(root=args.data_dir, 
                                         train=True, 
                                         download=True, 
                                         transform=train_transform
                                        )
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    model = eval(args.model)(num_classes=args.num_classes
                            ).to(device)
    optimiser = eval('optim.' + args.optim)(model.parameters(), 
                                            lr=args.lr, 
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay
                                           )
    if args.model == 'LeNet':
        scheduler = StepLR(optimiser, step_size=2, gamma=args.gamma)
    else:
        scheduler = CosineAnnealingLR(optimiser, T_max=200) 

    # track forgetting event
    idx2acc_list = defaultdict(list)
    last_batch = None
    pbar = tqdm(total=args.iterations, disable=not verbose)
    count = 0

    for idx in range(1, args.epochs+1):
        for i, (data, target, dataidx) in enumerate(train_loader):
            iter_idx = (idx - 1) * len(train_loader) + i
            pbar.update(1)
            if iter_idx > args.iterations: break

            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimiser.step()
            
            if last_batch is not None:
                last_data, last_target, last_dataidx, last_pred = last_batch
                new_pred = model(last_data).argmax(dim=1).detach()
                new_acc = (new_pred == last_target)
                old_acc = (last_pred == last_target)
                for k in range(len(last_dataidx)):
                    _count = 1 if old_acc[k].item() and not new_acc[k].item() \
                        else 0
                    count += _count
                    idx2acc_list[last_dataidx[k].item()].append(_count)
            
            last_batch = (data, target, dataidx, pred)
        if iter_idx > args.iterations: break
        scheduler.step()
    
    out_path = os.path.join(args.results_dir, f'{dataset}_{args.seed}.json')
    create_dir_for_file(out_path)
    with open(out_path, 'w') as f: json.dump(idx2acc_list, f)
    return count
