import os
import random
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import yaml

import wandb

def seed_everything(seed: int=42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_yaml(path: str):
    with open(path, "r") as f:
        load_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return load_yaml

def get_argumnets():
    '''
    example
    python train.py --config_yaml base.yaml
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, default='base_down_data.yaml')
    opts = parser.parse_args()
    return opts


def start_wandb(run_name):
    config = load_yaml('wandb\\wandb.yaml')
    wandb.init(entity=config['entity'],
            project="5th kakr",
            name=run_name)
