import os

import torch
from dataloader import CustomDataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import compute_pearson_correlation, load_yaml, seed_everything

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
prj_dir  = os.path.dirname(os.path.abspath(__file__))
