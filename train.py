import os

import torch
from dataloader import CustomDataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import load_yaml, seed_everything, get_argumnets

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
prj_dir  = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    config = get_argumnets()
    print(type(config.config))