import os
import random
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import transformers
from scipy.stats import pearsonr
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, default='base.yaml')
    opts = parser.parse_args()
    return opts


def compute_pearson_correlation(
    pred: transformers.trainer_utils.EvalPrediction,
) -> dict:
    """
    피어슨 상관 계수를 계산해주는 함수
        Args:
            pred (torch.Tensor): 모델의 예측값과 레이블을 포함한 데이터
        Returns:
            perason_correlation (dict): 입력값을 통해 계산한 피어슨 상관 계수
    """
    preds = pred.predictions.flatten()
    labels = pred.label_ids.flatten()
    perason_correlation = {"pearson_correlation": pearsonr(preds, labels)[0]}
    return perason_correlation

def start_wandb(run_name):
    config = load_yaml('wandb\\wandb.yaml')
    wandb.init(entity=config['entity'],
            project="5th kakr",
            name=run_name)