import torch
import torch.nn as nn

from transformers import BertModel


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, bidirectional, dropout, pretrained_embeddings=None):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [seq_len, batch_size]
        embedded = self.embedding(text)
        # embedded = [seq_len, batch_size, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded)
        # output = [seq_len, batch_size, hidden_dim * num_directions]
        # hidden = [num_layers * num_directions, batch_size, hidden_dim]
        # cell = [num_layers * num_directions, batch_size, hidden_dim]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.bidirectional else hidden[-1,:,:])
        # hidden = [batch_size, hidden_dim * num_directions]
        return self.fc(hidden)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, bidirectional, dropout, pretrained_embeddings=None):
        super(RNNClassifier, self).__init__()
        self.bidirectional = bidirectional
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [seq_len, batch_size]
        embedded = self.embedding(text)
        # embedded = [seq_len, batch_size, embedding_dim]
        output, hidden = self.rnn(embedded)
        # output = [seq_len, batch_size, hidden_dim * num_directions]
        # hidden = [num_layers * num_directions, batch_size, hidden_dim]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.bidirectional else hidden[-1,:,:])
        # hidden = [batch_size, hidden_dim * num_directions]
        return self.fc(hidden)
    

class BERTClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        logits = self.linear(pooled_output)
        return logits
    
class BERTClassifier1(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.linear(pooled_output)
        return logits