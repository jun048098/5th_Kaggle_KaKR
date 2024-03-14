import os

import torch
from dataloader import CustomDataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import load_yaml, seed_everything, get_argumnets, compute_pearson_correlation

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
prj_dir  = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    argumnets = get_argumnets()
    config_path = os.path.join(prj_dir, 'config_yaml', argumnets.config_yaml)
    config = load_yaml(path=config_path)
    seed_everything(config['seed'])

    print("cuda available:", torch.cuda.is_available())
    model = AutoModelForSequenceClassification.from_pretrained(
        config['architecture'],
        num_labels=1,
        ignore_mismatched_sizes=True
        )

    train_text_dataset = CustomDataset(
        data_file=config['data_folder']['train_data'],
        state='train',
        text_column='comment_text',
        target_column='toxicity',
        max_length=256,
        model_name=config['architecture']
    )

    args = TrainingArguments(
        output_dir=os.path.join(prj_dir, "save_folder", config["name"]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["n_epochs"],
        weight_decay=config["weight_decay"],
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        logging_steps=100,
        seed=config["seed"],
        group_by_length=True,
        lr_scheduler_type=config["scheduler"],
    )
    

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_text_dataset,
        eval_dataset=train_text_dataset,
        compute_metrics=compute_pearson_correlation,
    )

    trainer.train()
    

