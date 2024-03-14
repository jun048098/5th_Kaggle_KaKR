import os

import torch
from dataloader import CustomDataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import load_yaml, seed_everything, get_argumnets

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
prj_dir  = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    argumnets = get_argumnets()
    config_path = os.path.join(prj_dir, 'config_yaml', argumnets.config_yaml)
    config = load_yaml(path=config_path)
    seed_everything(config['seed'])

    model = AutoModelForSequenceClassification.from_pretrained(config['architecture'])

    train_text_dataset = CustomDataset(
        data_file=config['data_folder']['train_data'],
        state='train',
        text_column='comment_text',
        target_column='toxicity',
        max_length=256,
        model_name=config['architecture']
    )

    

    

