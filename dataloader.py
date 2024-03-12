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
        
        self.preprocess_data = self.preprocess(self.data)
        if self.state == 'test':
            self.input = self.tokenize_function(self.preprocess_data[text_column])
        else:
            self.input = self.tokenize_function(self.preprocess_data[text_column])
            self.target = self.preprocess_data[target_column]


    def __getitem__(self, idx):
        if self.state == "test":
            return {"input_ids": torch.tensor(self.inputs[idx])}
        else:
            if len(self.targets) == 0:
                return torch.tensor(self.inputs[idx])
            else:
                return {
                    "input_ids": torch.tensor(self.inputs[idx]),
                    "labels": torch.tensor(self.targets[idx]),
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
    

    def preprocess(self, df, text_column="comment_text"):
        url_pattern = r"https?://\S+|www\.\S+"
        # remove url
        df[text_column] = df[text_column].str.replace(url_pattern, " ")

        # apply unidecode
        df[text_column] = df[text_column].map(unidecode.unidecode)
        
        # remove emoji
        df[text_column] = df[text_column].map(self.remove_emoji)

        # apply lower
        df[text_column] = df[text_column].str.lower()
        
        return df


    def tokenize_function(self, data):
        tokenize_data = []
        for txt in tqdm(data, desc="Tokenizing", total=len(data)):
            output = self.tokenizer(
                txt,
                add_special_tokens= True,
                padding='max_length',
                truncation=True,
                max_length= self.max_length
            )
            tokenize_data.append(output)
        
        return tokenize_data
        