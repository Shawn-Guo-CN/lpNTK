import os
from munch import Munch
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from functorch import make_functional, make_functional_with_buffers
# from functorch import make_functional, vmap, vjp, jvp, jacrev

from utils import set_seed, create_dir_for_file, create_dir
from models import LeNet, ResNet18, ResNet50
from analysis.ntk import empirical_ntk


def approximate(config:Munch, dataset:str) -> None:
    args = config.default
    dataset_config = config[dataset]
    args.update(dataset_config)
    args.update({'experiment':config.wandb.experiment})

    use_cuda = args.use_cuda and torch.cuda.is_available()
    set_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False,
                       }
        train_kwargs.update(cuda_kwargs)

    transform = eval(dataset_config.transform)
    dataset1 = eval('datasets.'+dataset)(args.data_dir,
                                         train=True, 
                                         download=True, 
                                         transform=transform
                                        )
    data_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    data_iter = iter(data_loader)

    model = eval(args.model)(num_classes=args.num_classes).to(device)
    optimiser = eval('optim.' + args.optim)(model.parameters(), 
                                            lr=args.lr, 
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay
                                           )
    if args.model == 'LeNet':
        scheduler = StepLR(optimiser, step_size=2, gamma=args.gamma)
    else:
        scheduler = CosineAnnealingLR(optimiser, T_max=100)

    last_x = None
    last_y = None

    run = wandb.init(
              project=config.wandb.project, 
              name=config.wandb.experiment+'_'+dataset+'_'+str(args.seed),
              config=args,
              config_exclude_keys=["use_cuda", 
                                   "data_dir", 
                                   "iterations", 
                                   "log_interval", 
                                   "test_transform",
                                   "log_epochs",
                                   "log_iterations",
                                  ],
              entity=config.wandb.entity,
              reinit=True,
          )

    forget_tp_list = []
    forget_fp_list = []
    forget_tn_list = []
    forget_fn_list = []
    el2n_error_list = []
    num_counter = 0
    avg_precision = 0.
    avg_recall = 0.
    avg_f1 = 0.

    for iter_num in args.forget_iters:
        x, y = next(iter(data_iter))
        x, y = x.to(device), y.to(device)
        
        if last_x is None:
            last_x = x
            last_y = y
            scheduler.step()
            continue
        
        # load the checkpoint of the previous step
        last_checkpoint_path = os.path.join(args.checkpoints_dir,
                                       dataset,
                                       str(args.seed),
                                       'iterations',
                                       f'{args.model}_{iter_num-1}.pt'
                                      )
        model.load_state_dict(torch.load(last_checkpoint_path))
        model.eval()
        if args.model == 'LeNet':
            fnet, params = make_functional(model)
            def fnet_single(params, x):
                return fnet(params, x.unsqueeze(0)).squeeze(0)
        else:
            fnet, params, buffers = make_functional_with_buffers(model)
            def fnet_single(params, x):
                return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

        # ground truth of the current sample
        y_onehot = F.one_hot(y, num_classes=args.num_classes).to(device)
        last_y_onehot = F.one_hot(last_y, 
                                  num_classes=args.num_classes
                                 ).to(device)
        # prediction of the current sample
        pred_x = F.softmax(model(x), dim=1).to(device)
        pred_last_x = F.softmax(model(last_x), dim=1).to(device)
        # prediction error
        pred_error_x = y_onehot - pred_x

        # approximate the change of predictions of the current x 
        # with parameters from the previous time-step
        deltas = torch.zeros_like(pred_last_x)
        lr = scheduler._last_lr[0]

        for i in range(last_x.shape[0]):
            A_matrix = torch.diag(pred_last_x[i]) - \
                       torch.outer(pred_last_x[i], pred_last_x[i])
            ntk_matrix = empirical_ntk(fnet_single, params,
                                       last_x[i].unsqueeze(0), x
                                      ).detach()
            A_times_ntk = torch.einsum('ij, abjk -> abik', A_matrix, ntk_matrix)
            deltas[i] = lr * torch.einsum('abij, bj->abi', 
                                          A_times_ntk, pred_error_x
                                         ).mean(dim=1)[0]

        # approximated prediction of the previous x at the current time-step
        pred_last_xt_approx = pred_last_x + deltas
        
        # load the checkpoint of the current time step
        checkpoint_path = os.path.join(args.checkpoints_dir,
                                       dataset,
                                       str(args.seed),
                                       'iterations',
                                       f'{args.model}_{iter_num}.pt'
                                      )
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        # true prediction of the last sample
        pred_last_xt_true = F.softmax(model(last_x), dim=1).to(device)
        
        # true forgetting indicator
        forget_indicator_true = torch.logical_and(
            (torch.argmax(pred_last_x, dim=1) == last_y),
            torch.logical_not(
                (torch.argmax(pred_last_xt_true, dim=1) == last_y)
            )
        )
        
        # approximate forgetting indicator
        forget_indicator_approx = torch.logical_and(
            (torch.argmax(pred_last_x, dim=1) == last_y),
            torch.logical_not(
                (torch.argmax(pred_last_xt_approx, dim=1) == last_y)
            )
        )

        # true positive =>
        # true in forget_indicator_approx and true in forget_indicator_true
        tp = torch.logical_and(
                               forget_indicator_approx, forget_indicator_true
                              ).int().sum().cpu().item()
        # false positive =>
        # true in forget_indicator_approx and false in forget_indicator_true
        fp = torch.logical_and(forget_indicator_approx,
                               torch.logical_not(forget_indicator_true)
                              ).int().sum().cpu().item()
        # true negative =>
        # false in forget_indicator_approx and false in forget_indicator_true
        tn = torch.logical_and(torch.logical_not(forget_indicator_approx),
                               torch.logical_not(forget_indicator_true)
                              ).int().sum().cpu().item()
        # false negative =>
        # false in forget_indicator_approx and true in forget_indicator_true
        fn = torch.logical_and(torch.logical_not(forget_indicator_approx),
                               forget_indicator_true
                              ).int().sum().cpu().item()
        
        forget_tp_list.append(tp)
        forget_fp_list.append(fp)
        forget_tn_list.append(tn)
        forget_fn_list.append(fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.

        avg_precision = \
            (avg_precision * num_counter + precision) / (num_counter + 1)
        avg_recall = \
            (avg_recall * num_counter + recall) / (num_counter + 1)
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        num_counter += 1

        log_table = wandb.Table(
            data=[
                ['Precision', avg_precision],
                ['Recall', avg_recall],
                ['F1', avg_f1],
            ],
            columns=["metric", "value"]
        )
        run.log({"exp1_result" : wandb.plot.bar(log_table, "metric",
           "value", title="Performance on predicting forgetting events.")
        })

        # true el2n of the last sample
        el2n_last_xt_true = torch.norm((pred_last_xt_true - last_y_onehot),
                                       p=2,
                                       dim=1
                                      )
        # approximate el2n of the last sample
        el2n_last_xt_approx = torch.norm((pred_last_xt_approx - last_y_onehot),
                                         p=2,
                                         dim=1
                                        )
        el2n_error_abs = ((el2n_last_xt_approx / el2n_last_xt_true) - 1).abs()
        run.log({'exp2_result': el2n_error_abs.mean().cpu().item()})
        el2n_error_list.append(el2n_error_abs.mean().cpu().item())
        last_x = x
        last_y = y
        scheduler.step()

    log_file_path = \
        os.path.join(args.logs_dir, dataset, str(args.seed), 'log.csv')
    create_dir_for_file(log_file_path)
    print(len(args.forget_iters))
    print(len(forget_tp_list))
    print(len(el2n_error_list))
    with open(log_file_path, 'w') as f:
        f.write('iter,tp,fp,tn,fn,el2n_error\n')
        for idx, iter_num in enumerate(args.forget_iters[:-1]):
            f.write(f'{iter_num+1},'+
                    f'{forget_tp_list[idx]},{forget_fp_list[idx]},'+
                    f'{forget_tn_list[idx]},{forget_fn_list[idx]},'+
                    f'{el2n_error_list[idx]}\n'
                   )
    
    run.finish()
