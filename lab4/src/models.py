import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
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
        return self.fc(hidden).sigmoid()


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super(RNNClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [seq_len, batch_size]
        embedded = self.embedding(text)
        # embedded = [seq_len, batch_size, embedding_dim]
        output, hidden = self.rnn(embedded)
        # output = [seq_len, batch_size, hidden_dim * num_directions]
        # hidden = [num_layers * num_directions, batch_size, hidden_dim]
        # cell = [num_layers * num_directions, batch_size, hidden_dim]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.bidirectional else hidden[-1,:,:])
        # hidden = [batch_size, hidden_dim * num_directions]
        return self.fc(hidden).sigmoid()