import os, sys
import toml
import argparse
from munch import Munch, munchify
from collections import defaultdict
import pickle


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp5")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from analysis.retrain_models import retrain
from utils import update_config, create_dir_for_file


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.results_dir = RESULTS_DIR

    def _retrain_on_dataset_(config:Munch, dataset:str) -> None:
        dataset2testacc_list = defaultdict(list)

        for seed in config.sweep.seeds:
            config.default.seed = seed
            for pruned_dataset in config.sweep.pruned_datasets:
                if pruned_dataset == 'raw' or 'selfmax' in pruned_dataset:
                    ratio_list = [1.0]
                else:
                    ratio_list = config.sweep.ratios
                for pruned_ratio in ratio_list:
                    config.default.pruned_set = pruned_dataset
                    config.default.ratio = pruned_ratio
                    test_acc, pruned_set = retrain(config, dataset)
                    dataset2testacc_list[pruned_set].append(test_acc)
        
        out_fname = os.path.join(RESULTS_DIR,
                                 dataset,
                                 f"testacc_cmp_results.pkl"
                                )
        create_dir_for_file(out_fname)
        with open(out_fname, "wb") as f:
            pickle.dump(dataset2testacc_list, f)

    if args.mnist:
        _retrain_on_dataset_(config, "MNIST")
    
    if args.cifar10:
        _retrain_on_dataset_(config, "CIFAR10")

    if args.cifar100:
        _retrain_on_dataset_(config, "CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="ReTraining")
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    config = toml.load(os.path.join(SCRIPTS_DIR,"exp5", "config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)

    run(args, config)