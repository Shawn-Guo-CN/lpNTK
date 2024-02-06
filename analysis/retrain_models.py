import os
from random import shuffle
from munch import Munch
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from utils import set_seed, create_dir_for_file, create_dir
from models import LeNet, ResNet18, ResNet50
from pruned_datasets import PrunedMNIST, PrunedCIFAR10

from analysis.pretrain_models import test


def get_pruned_dataset_fname(args:Munch, dataset:str) -> str:
    return f'{args.pruned_set}_{args.ratio}.npy'


def retrain(config:Munch, dataset:str):
    args = config.default
    dataset_config = config[dataset]
    args.update(dataset_config)
    args.update({'experiment':config.wandb.experiment})
    
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
    test_transform = eval(dataset_config.test_transform)

    train_set = None
    if args.pruned_set == 'raw':
        train_set = eval('datasets.'+dataset)(args.data_dir, 
                                             train=True, 
                                             download=True, 
                                             transform=train_transform
                                            )
        args.pruned_set = dataset
    else:
        dataset_fname = get_pruned_dataset_fname(args, dataset)
        train_set = eval(f'Pruned{dataset}')(
                        os.path.join(args.data_dir,
                                     dataset,
                                     'pruned',
                                     dataset_fname,
                        ),
                        transform = train_transform
                    )
        args.pruned_set = dataset_fname[:-4]
    test_set = eval('datasets.'+dataset)(args.data_dir, 
                                         train=False,
                                         download=True,
                                         transform=test_transform
                                        )
    
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                              shuffle=False, 
                                              pin_memory=True
                                             )

    model = eval(args.model)(num_classes=args.num_classes).to(device)
    optimiser = eval('optim.' + args.optim)(model.parameters(), 
                                            lr=args.lr, 
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay
                                           )
    
    if args.model == 'LeNet':
        scheduler = StepLR(optimiser, step_size=2, gamma=args.gamma)
    else:
        scheduler = CosineAnnealingLR(optimiser, T_max=200)

    run = wandb.init(project=config.wandb.project, 
                     name=config.wandb.experiment+dataset+'_'+str(args.seed),
                     config=args,
                     config_exclude_keys=["use_cuda", 
                                          "data_dir", 
                                          "iterations",
                                         ],
                     entity="None", # replace with your wandb entity
                     reinit=True,
                    )

    best_test_acc = -1

    test_acc = test(model, device, test_loader)
    run.log({config.wandb.experiment+" test acc": test_acc})

    for _ in range(args.epochs):
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimiser.step()
            run.log({config.wandb.experiment+" loss": loss.item()})
        
        test_acc = test(model, device, test_loader)
        run.log({config.wandb.experiment+" test acc": test_acc})
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        scheduler.step()
    
    run.finish()

    return best_test_acc, args.pruned_set