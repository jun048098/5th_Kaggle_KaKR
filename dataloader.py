import re
import unidecode

import pandas as pd
from tqdm.auto import tqdm
import torch

from transformers import AutoTokenizer


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_file,
            state,
            text_column = None,
            target_column = None,
            max_length = 512,
            model_name = "bert-base-uncased",
    ):
        self.data = pd.read_csv(data_file)
        self.state = state
        self.max_length = max_length
        self.tokenizer =  AutoTokenizer.from_pretrained(model_name)
        
        self.preprocess_data = self.text_preprocess(self.data)
        if self.state == 'test':
            self.inputs = self.tokenize_function(self.preprocess_data[text_column])
        else:
            self.inputs = self.tokenize_function(self.preprocess_data[text_column])
            self.targets = self.preprocess_data[target_column]


    def __getitem__(self, idx):
        if self.state == "test":
            return {
                    "input_ids": torch.tensor(self.inputs["input_ids"][idx]),
                    "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
                    }
        else:
            if len(self.targets) == 0:
                return {
                    "input_ids": torch.tensor(self.inputs["input_ids"][idx]),
                    "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
                    }
            else:
                return {
                    "input_ids": torch.tensor(self.inputs["input_ids"][idx]),
                    "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
                    "label": torch.tensor(self.targets[idx]),
                    }
                    

    def __len__(self):
        return len(self.data)
    

    def remove_emoji(self, string):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)
    

    def text_preprocess(self, data, text_column="comment_text"):
        url_pattern = r"https?://\S+|www\.\S+"
        # remove url
        data[text_column] = data[text_column].str.replace(url_pattern, " ")

        # apply unidecode
        data[text_column] = data[text_column].map(unidecode.unidecode)
        
        # remove emoji
        data[text_column] = data[text_column].map(self.remove_emoji)

        # apply lower
        data[text_column] = data[text_column].str.lower()
        
        return data


    def tokenize_function(self, data):
        tokenize_data = {
            "input_ids": [],
            "attention_mask": []
        }

        for txt in tqdm(data, desc="Tokenizing", total=len(data)):
            output = self.tokenizer(
                txt,
                add_special_tokens= True,
                padding='max_length',
                truncation=True,
                max_length= self.max_length
            )
            tokenize_data["input_ids"].append(output["input_ids"])
            tokenize_data["attention_mask"].append(output["attention_mask"])
        
        return tokenize_data
        