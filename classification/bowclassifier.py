import torch.nn as nn
import torch.nn.functional as F

class BoWClassifier(nn.Module):

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vector):
        return F.log_softmax(self.linear(bow_vector))