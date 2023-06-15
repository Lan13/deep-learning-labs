import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from tqdm.auto import tqdm


class CustomTextDataset1(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        numericalized_text = self.numericalize_text(text)
        return numericalized_text, label

    def __len__(self):
        return len(self.labels)
    
    def numericalize_text(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]
     

class CustomTextDataset2(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(self.texts, truncation=True, padding=True)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


# preprocess the data with a collate function, and pads the input sequences to the maximum length in the batch:
def collate_batch(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        # processed_text = torch.tensor(_text)
        text_list.append(torch.tensor(_text))
    padded_text = pad_sequence(text_list, batch_first=False, padding_value=1.0)
    return torch.tensor(label_list, dtype=torch.float64).to(device), padded_text.to(device)


def read_data(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels
