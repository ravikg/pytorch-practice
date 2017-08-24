import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, lstm_gru = 'LSTM', n_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if lstm_gru == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim)
        else:
            self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result