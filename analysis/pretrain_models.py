import os
from random import shuffle
from munch import Munch
import wandb

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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, 
                                         reduction='sum'
                                        ).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)  
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    model.train()

    return 100. * correct / len(test_loader.dataset)


def pretrain(config:Munch, dataset:str, best_val:float=-0.1):
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
    dataset1 = eval('datasets.'+dataset)(root=args.data_dir, 
                                         train=True, 
                                         download=True, 
                                         transform=train_transform
                                        )
    dataset2 = eval('torchvision.datasets.'+dataset)(root=args.data_dir, 
                                         train=False, 
                                         transform=test_transform
                                        )
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, shuffle=False, pin_memory=True)

    model = eval(args.model)(num_classes=dataset_config.num_classes
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
    
    best_test_acc = best_val

    run = wandb.init(project=config.wandb.project, 
                     name=config.wandb.experiment+dataset+'_'+str(args.seed),
                     config=args,
                     config_exclude_keys=["use_cuda", 
                                          "data_dir", 
                                          "iterations", 
                                          "log_interval", 
                                          "test_transform",
                                          "log_epochs",
                                          "log_iterations",
                                         ],
                     entity="None", # replace with your wandb entity
                     reinit=True,
                    )

    loss_list = []
    testacc_list = []
    checkpoint_dir = os.path.join(args.results_dir, dataset, str(args.seed))
    create_dir(checkpoint_dir)
    chkpoint_iter_dir = \
        os.path.join(args.results_dir, dataset,  str(args.seed), 'iterations')
    create_dir(chkpoint_iter_dir)
    chkpoint_epoch_idr = \
        os.path.join(args.results_dir, dataset,  str(args.seed), 'epochs')
    create_dir(chkpoint_epoch_idr)
    
    torch.save(model.state_dict(), 
               os.path.join(checkpoint_dir, f"{args.model}_init.pt")
              )
    ld_track = np.zeros((len(dataset1), args.epochs)) 
    # track learning difficulty

    for idx in range(1, args.epochs+1):
        for i, (data, target, dataidx) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            l2 = get_l2distance(output, target)
            loss.backward()
            optimiser.step()
            run.log({config.wandb.experiment+"loss": loss.item()})

            iter_idx = (idx - 1) * len(train_loader) + i
            if iter_idx in args.log_iterations:
                torch.save(model.state_dict(),
                           os.path.join(chkpoint_iter_dir,
                                        args.model+'_'+str(iter_idx)+'.pt'
                                       )
                          )
            ld_track[dataidx ,idx-1] = l2
            
        if idx % args.log_interval == 0:
            test_acc = test(model, device, test_loader)
            run.log({config.wandb.experiment+"test acc": test_acc})
            loss_list.append(loss.item())
            testacc_list.append(test_acc)

            if best_test_acc < test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 
                           os.path.join(os.path.join(args.results_dir, 
                                                     dataset), 
                                        f"{args.model}_best.pt"
                                       )
                          )
        
        if idx in args.log_epochs:
            torch.save(model.state_dict(), 
                       os.path.join(chkpoint_epoch_idr,
                                    f"{args.model}_{idx}.pt"
                                   )
                      )

        scheduler.step()
    run.finish()

    log_file_path = \
        os.path.join(args.logs_dir, dataset, str(args.seed), 'log.csv')
    create_dir_for_file(log_file_path)
    with open(log_file_path, 'w') as f:
        f.write('iter,loss,test_acc\n')
        for idx in range(len(loss_list)):
            f.write(f'{idx+1},{loss_list[idx]},{testacc_list[idx]}\n')
    
    ld_track_path = \
        os.path.join(args.logs_dir, dataset, str(args.seed), 'ld_track.npy')
    np.save(ld_track_path, ld_track, allow_pickle=True)

    return best_test_acc


if __name__ == '__main__':
    pretrain()
