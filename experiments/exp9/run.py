import os, sys
import toml
import argparse
from munch import munchify


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
PT_DIR = os.path.join(PROJ_DIR, "checkpoints")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp1")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
RESULTS_DIR = os.path.join(PROJ_DIR, "data")

sys.path.append(PROJ_DIR)
from analysis.ntk import get_ntk
from utils import update_config


def run(args, config):
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR

    if args.mnist:
        config.default.result_path = os.path.join(RESULTS_DIR, "MNIST")
        config.default.pt_file = \
            os.path.join(PT_DIR, f"MNIST_{config.MNIST.model}.pt")
        get_ntk(config, "MNIST")
    
    if args.cifar10:
        config.default.result_path = os.path.join(RESULTS_DIR, "CIFAR10")
        config.default.pt_file = \
            os.path.join(PT_DIR, f"CIFAR10_{config.CIFAR10.model}.pt")
        get_ntk(config, "CIFAR10")

    if args.cifar100:
        config.default.result_path = os.path.join(RESULTS_DIR, "CIFAR10")
        config.default.pt_file = \
            os.path.join(PT_DIR, f"CIFAR100_{config.CIFAR100.model}.pt")
        get_ntk(config, "CIFAR100")


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

    config = toml.load(os.path.join(SCRIPTS_DIR, "exp1", "config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)

    run(args, config)

