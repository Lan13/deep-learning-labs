import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Define a custom dataset class for the text data
class CustomTextDataset(Dataset):
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

# preprocess the data with a collate function, and pads the input sequences to the maximum length in the batch:
def collate_batch(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text)
        text_list.append(processed_text)
    padded_text = pad_sequence(text_list, batch_first=False, padding_value=1.0)
    return torch.tensor(label_list, dtype=torch.float64).to(device), padded_text.to(device)

