import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text_file_path, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(text_file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        tokenized_output = self.tokenizer(
            line,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return tokenized_output['input_ids'].squeeze(0)