import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer_=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer_
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        item = {
            key: val.squeeze(0)
            for key, val in encodings.items()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item