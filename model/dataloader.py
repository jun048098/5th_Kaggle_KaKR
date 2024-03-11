import pandas as pd
from tqdm.auto import tqdm
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_file
    ):
        self.data = pd.read_csv(data_file)

    def __getitem__(self, idx):
        return
    
    def __len__(self):
        return len(self.data)