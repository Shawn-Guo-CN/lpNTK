import os, sys
import toml
import argparse
from munch import Munch, munchify


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp6")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from analysis.forget_el2n import approximate
from utils import update_config


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.checkpoints_dir = CHECKPOINTS_DIR
    config.default.results_dir = RESULTS_DIR
    
    def _approximate_on_dataset_(dataset:str) -> None:
        for seed in config.sweep.seeds:
            config.default.seed = seed
            approximate(config, dataset)

    if args.mnist:
        _approximate_on_dataset_("MNIST")

    if args.cifar10:
        _approximate_on_dataset_("CIFAR10")

    if args.cifar100:
        _approximate_on_dataset_("CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    config = toml.load(os.path.join(SCRIPTS_DIR, "exp6", "config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)

    run(args, config)