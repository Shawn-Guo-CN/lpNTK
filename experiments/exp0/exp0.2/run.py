import os, sys
import toml
import argparse
from munch import Munch, munchify


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp0")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
RESULTS_DIR = os.path.join(PROJ_DIR, "data")

sys.path.append(PROJ_DIR)
from analysis.ntk import get_ntk
from utils import update_config


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.results_dir = RESULTS_DIR

    if args.mnist:
        get_ntk(config, 'MNIST')
    
    if args.cifar10:
        get_ntk(config, 'CIFAR10')

    if args.cifar100:
        get_ntk(config, 'CIFAR100')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="PreTraining")
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
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    config = toml.load(os.path.join(SCRIPTS_DIR,"exp0","exp0.2","config.toml"))
    config = update_config(unknown, config)
    config['default']['start_idx'] = args.start_idx
    config['default']['end_idx'] = args.end_idx
    config = munchify(config)

    run(args, config)