import os, sys
import toml
import argparse
from munch import Munch, munchify


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp0", "exp0.1")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
RESULTS_DIR = os.path.join(PROJ_DIR, "checkpoints")

sys.path.append(PROJ_DIR)
from analysis.pretrain_models import pretrain
from utils import update_config


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.results_dir = RESULTS_DIR

    def _pretrain_on_dataset_(dataset:str) -> None:
        best_test_acc = -1.0
        for seed in config.sweep.seeds:
            config.default.seed = seed
            test_acc = pretrain(config, dataset, best_val=best_test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
        print(f"On {dataset}, best test acc: {best_test_acc}")

    if args.mnist:
        _pretrain_on_dataset_("MNIST")
    
    if args.cifar10:
        _pretrain_on_dataset_("CIFAR10")

    if args.cifar100:
        _pretrain_on_dataset_("CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="PreTraining")
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    config = toml.load(os.path.join(SCRIPTS_DIR,"exp0","exp0.1","config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)

    run(args, config)