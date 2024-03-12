import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import yaml


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