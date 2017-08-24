import torch.nn as nn
import torch
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_gru = 'LSTM', n_layers=1):
        super(EncoderRNN, self).__init__()

        self.lstm_gru = lstm_gru
        self.n_layers = n_layers
        self.hidden_size = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if self.lstm_gru == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim)
        else:
            self.rnn = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        if self.lstm_gru == 'GRU':
            result = Variable(torch.zeros(1, 1, self.hidden_size))
        else:
            result = (Variable(torch.zeros(1, 1, self.hidden_size)),
                      Variable(torch.zeros(1, 1, self.hidden_size)))
        return result
